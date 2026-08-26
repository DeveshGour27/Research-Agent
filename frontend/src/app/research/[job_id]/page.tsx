"use client";

import * as React from "react";
import { useParams } from "next/navigation";
import { Badge } from "@/components/ui/badge";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Spinner } from "@/components/ui/spinner";
import { Alert } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Dialog } from "@/components/ui/dialog";
import { Activity, Clock, CheckCircle2, XCircle, Beaker, PlayCircle, Loader2, User, FileText } from "lucide-react";
import { ResearchSSEController, ConnectionState, ResearchEvent } from "@/lib/api/events";
import { getHitlRequests, approveHitlRequest, rejectHitlRequest, HITLRequestResponse } from "@/lib/api/hitl";
import { getResearchResult, ResearchResultResponse } from "@/lib/api/results";
import { getResearchJobStatus } from "@/lib/api/research";
import { Markdown } from "@/components/ui/markdown";

function getStatusColor(status: string) {
  switch (status.toUpperCase()) {
    case "PENDING":
    case "RUNNING":
      return "default";
    case "WAITING_FOR_HUMAN":
      return "warning";
    case "COMPLETED":
      return "success";
    case "FAILED":
      return "error";
    case "CANCELLED":
      return "neutral";
    default:
      return "neutral";
  }
}

function getEventIcon(eventType: string) {
  if (eventType.includes("STARTED")) return <PlayCircle className="h-4 w-4 text-primary" />;
  if (eventType.includes("COMPLETED")) return <CheckCircle2 className="h-4 w-4 text-success" />;
  if (eventType.includes("FAILED")) return <XCircle className="h-4 w-4 text-error" />;
  if (eventType.includes("HITL_REQUESTED")) return <User className="h-4 w-4 text-warning" />;
  if (eventType.includes("HITL_APPROVED")) return <CheckCircle2 className="h-4 w-4 text-success" />;
  if (eventType.includes("HITL_REJECTED")) return <XCircle className="h-4 w-4 text-error" />;
  if (eventType.includes("TOOL")) return <Activity className="h-4 w-4 text-secondary-foreground" />;
  return <Beaker className="h-4 w-4 text-secondary-foreground" />;
}

export default function JobDashboardPage() {
  const params = useParams();
  const rawJobId = params?.job_id;
  const jobId = Array.isArray(rawJobId) ? rawJobId[0] : rawJobId;

  const [connectionState, setConnectionState] = React.useState<ConnectionState>("DISCONNECTED");
  const [events, setEvents] = React.useState<ResearchEvent[]>([]);
  const [jobStatus, setJobStatus] = React.useState("CONNECTING...");
  
  // HITL State
  const [pendingHitl, setPendingHitl] = React.useState<HITLRequestResponse | null>(null);
  const [isSubmittingHitl, setIsSubmittingHitl] = React.useState(false);
  const [showRejectDialog, setShowRejectDialog] = React.useState(false);
  const [hitlError, setHitlError] = React.useState<string | null>(null);

  // Result State
  const [result, setResult] = React.useState<ResearchResultResponse | null>(null);
  const [isFetchingResult, setIsFetchingResult] = React.useState(false);
  const [resultError, setResultError] = React.useState<string | null>(null);
  
  const fetchResult = React.useCallback(async () => {
    if (!jobId) return;
    setIsFetchingResult(true);
    setResultError(null);
    try {
      const res = await getResearchResult(jobId);
      setResult(res);
    } catch (err: unknown) {
      const error = err as Error;
      setResultError(error.message || "Failed to load research result");
    } finally {
      setIsFetchingResult(false);
    }
  }, [jobId]);

  const fetchPendingHitl = React.useCallback(async () => {
    if (!jobId) return;
    try {
      const data = await getHitlRequests(jobId);
      const pending = data.hitl_requests.find((r) => r.status === "PENDING");
      setPendingHitl(pending || null);
      if (pending) {
        setJobStatus((prev) => (prev !== "WAITING_FOR_HUMAN" ? "WAITING_FOR_HUMAN" : prev));
      }
    } catch (err) {
      console.error("Failed to fetch HITL requests:", err);
    }
  }, [jobId]);

  // Initial Direct Page Load Sync
  React.useEffect(() => {
    if (!jobId) return;
    getResearchJobStatus(jobId).then((statusRes) => {
      setJobStatus(statusRes.status);
      if (statusRes.status === "COMPLETED") {
        fetchResult();
      }
    }).catch(console.error);
  }, [jobId, fetchResult]);

  React.useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    fetchPendingHitl();
  }, [fetchPendingHitl]);

  React.useEffect(() => {
    if (!jobId) return;

    const controller = new ResearchSSEController({
      jobId,
      onEvent: (event) => {
        setEvents((prev) => {
          if (prev.some((e) => e.event_id === event.event_id)) return prev;
          
          const newEvents = [...prev, event];
          newEvents.sort((a, b) => {
            if (a.sequence !== undefined && b.sequence !== undefined) {
              return a.sequence - b.sequence;
            }
            return new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime();
          });
          return newEvents;
        });

        if (event.event_type === "JOB_CREATED") setJobStatus("PENDING");
        if (event.event_type === "JOB_STARTED") setJobStatus("RUNNING");
        if (event.event_type === "JOB_FAILED") setJobStatus("FAILED");
        if (event.event_type === "JOB_CANCELLED") setJobStatus("CANCELLED");
        if (event.event_type === "JOB_COMPLETED") {
          setJobStatus("COMPLETED");
          fetchResult();
        }
        
        if (event.event_type.startsWith("HITL_")) {
          fetchPendingHitl();
        }
      },
      onStateChange: (state) => {
        setConnectionState(state);
      },
    });

    controller.connect();

    return () => {
      controller.disconnect();
    };
  }, [jobId, fetchPendingHitl, fetchResult]);

  const handleApprove = async () => {
    if (!jobId || !pendingHitl) return;
    setHitlError(null);
    setIsSubmittingHitl(true);
    try {
      await approveHitlRequest(jobId, pendingHitl.request_id);
      setPendingHitl(null);
      setJobStatus("RUNNING");
    } catch (err: unknown) {
      const error = err as Error;
      setHitlError(error.message || "Failed to approve request.");
    } finally {
      setIsSubmittingHitl(false);
    }
  };

  const handleReject = async () => {
    if (!jobId || !pendingHitl) return;
    setHitlError(null);
    setIsSubmittingHitl(true);
    setShowRejectDialog(false);
    try {
      await rejectHitlRequest(jobId, pendingHitl.request_id);
      setPendingHitl(null);
      setJobStatus("RUNNING");
    } catch (err: unknown) {
      const error = err as Error;
      setHitlError(error.message || "Failed to reject request.");
    } finally {
      setIsSubmittingHitl(false);
    }
  };

  if (!jobId) {
    return <Alert variant="error" title="Error">Invalid Job ID</Alert>;
  }

  const renderConnectionBadge = () => {
    if (jobStatus === "WAITING_FOR_HUMAN" && connectionState === "CONNECTED") {
      return <Badge variant="success">● Connected</Badge>;
    }
    switch (connectionState) {
      case "CONNECTED":
        return <Badge variant="success">● Connected</Badge>;
      case "CONNECTING":
      case "RECONNECTING":
        return <Badge variant="neutral"><Loader2 className="h-3 w-3 mr-1 animate-spin inline" /> {connectionState}</Badge>;
      case "DISCONNECTED":
        return <Badge variant="error">● Disconnected</Badge>;
      case "FAILED":
        return <Badge variant="error">Connection Failed</Badge>;
      case "COMPLETED":
      case "CLOSED":
        return <Badge variant="neutral">Stream Closed</Badge>;
    }
  };

  const renderResultSection = () => {
    if (jobStatus === "FAILED") {
      return (
        <Alert variant="error" title="Research Failed">
          {result?.error || "The research job encountered a fatal error and could not complete."}
        </Alert>
      );
    }
    
    if (jobStatus !== "COMPLETED") return null;

    if (isFetchingResult) {
      return (
        <Card className="border-primary/20">
          <CardContent className="flex flex-col items-center justify-center py-12 space-y-4">
            <Spinner />
            <p className="text-secondary-foreground">Preparing your research report...</p>
          </CardContent>
        </Card>
      );
    }

    if (resultError) {
      return (
        <Alert variant="error" title="Unable to load the research result">
          <p className="mb-4">{resultError}</p>
          <Button variant="secondary" size="sm" onClick={fetchResult}>Retry</Button>
        </Alert>
      );
    }

    if (!result?.result) {
      return (
        <Alert variant="info" title="Research completed">
          The research job completed successfully, but no report is currently available.
        </Alert>
      );
    }

    return (
      <Card className="border-border">
        <CardHeader className="border-b border-border bg-surface-elevated/50">
          <CardTitle className="text-xl flex items-center gap-2">
            <FileText className="h-5 w-5 text-primary" />
            Research Result
          </CardTitle>
        </CardHeader>
        <CardContent className="pt-6">
          <Markdown content={result.result} />
        </CardContent>
      </Card>
    );
  };

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      {/* Header Section */}
      <div className="flex flex-col md:flex-row md:items-start justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Research Job</h1>
          <p className="text-secondary-foreground font-mono text-sm mt-1">{jobId}</p>
        </div>
        <div className="flex items-center gap-3">
          {renderConnectionBadge()}
          <Badge variant={getStatusColor(jobStatus)} className="text-sm px-3 py-1 uppercase">
            {jobStatus === "WAITING_FOR_HUMAN" ? "WAITING FOR HUMAN" : jobStatus}
          </Badge>
        </div>
      </div>

      {connectionState === "FAILED" && jobStatus !== "COMPLETED" && (
        <Alert variant="error" title="Connection Interrupted">
          Unable to maintain the live connection to the research agent. 
          Please refresh the page to try again.
        </Alert>
      )}

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 flex-col-reverse lg:flex-row">
        
        {/* Left/Main Column: Result & Event Timeline */}
        <div className="lg:col-span-2 space-y-6">
          
          {/* Result Output */}
          {renderResultSection()}

          {/* Event Timeline */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Event Timeline</CardTitle>
            </CardHeader>
            <CardContent>
              {events.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-12 text-secondary-foreground space-y-4">
                  {connectionState === "CONNECTING" || connectionState === "RECONNECTING" ? (
                    <>
                      <Spinner />
                      <p>Connecting to research job...</p>
                    </>
                  ) : (
                    <p>No events received yet.</p>
                  )}
                </div>
              ) : (
                <div className="relative border-l border-border ml-4 space-y-6">
                  {events.map((ev) => (
                    <div key={ev.event_id} className="relative pl-6">
                      <div className="absolute -left-[9px] top-1 bg-background rounded-full p-0.5 border border-border">
                        {getEventIcon(ev.event_type)}
                      </div>
                      
                      <div className="space-y-1">
                        <div className="flex items-center gap-2">
                          <span className="font-semibold text-foreground text-sm">
                            {ev.event_type}
                          </span>
                          <span className="text-xs text-secondary-foreground flex items-center gap-1">
                            <Clock className="h-3 w-3" />
                            {new Date(ev.timestamp).toLocaleTimeString()}
                          </span>
                        </div>
                        
                        <div className="text-sm text-secondary-foreground">
                          {typeof ev.payload?.goal === "string" && <p>Goal: {ev.payload.goal}</p>}
                          {typeof ev.payload?.error === "string" && <p className="text-error font-medium">{ev.payload.error}</p>}
                          {typeof ev.payload?.tool_name === "string" && <p>Tool: {ev.payload.tool_name}</p>}
                          {typeof ev.payload?.component_name === "string" && <p>Component: {ev.payload.component_name}</p>}
                          {typeof ev.payload?.reason === "string" && <p>Reason: {ev.payload.reason}</p>}
                          
                          <details className="mt-2 text-xs">
                            <summary className="cursor-pointer text-primary hover:underline select-none inline-block">
                              Event details ▾
                            </summary>
                            <div className="mt-2 p-2 bg-muted rounded-md overflow-x-auto border border-border">
                              <pre className="text-secondary-foreground whitespace-pre-wrap break-all">
                                {JSON.stringify(ev.payload, null, 2)}
                              </pre>
                            </div>
                          </details>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Metadata Sidebar & HITL Request (Right Column) */}
        <div className="space-y-6">
          {pendingHitl && (
            <Card className="border-warning">
              <CardHeader className="bg-warning/10 border-b border-warning/20">
                <CardTitle className="text-lg text-warning flex items-center gap-2">
                  <User className="h-5 w-5" />
                  Human Review Required
                </CardTitle>
              </CardHeader>
              <CardContent className="pt-4 space-y-4">
                <p className="text-sm text-secondary-foreground">
                  The research agent requires your decision to proceed.
                </p>
                
                {hitlError && (
                  <Alert variant="error" title="Error">{hitlError}</Alert>
                )}

                <div className="space-y-2">
                  <div className="bg-muted p-3 rounded-md text-sm border border-border">
                    <p><span className="font-semibold">Type:</span> {pendingHitl.request_type}</p>
                    <p><span className="font-semibold">Component:</span> {pendingHitl.component_name}</p>
                    <p className="text-xs text-secondary-foreground mt-2">
                      Requested: {new Date(pendingHitl.created_at).toLocaleString()}
                    </p>
                  </div>
                </div>

                <div className="flex flex-col gap-2 pt-2">
                  <Button 
                    variant="default" 
                    className="w-full" 
                    onClick={handleApprove}
                    disabled={isSubmittingHitl}
                  >
                    {isSubmittingHitl ? <><Loader2 className="mr-2 h-4 w-4 animate-spin" /> Submitting...</> : "Approve"}
                  </Button>
                  <Button 
                    variant="danger" 
                    className="w-full" 
                    onClick={() => setShowRejectDialog(true)}
                    disabled={isSubmittingHitl}
                  >
                    Reject
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}

          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Details</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <p className="text-xs text-secondary-foreground mb-1">Started</p>
                <p className="text-sm font-medium">
                  {events.length > 0 ? new Date(events[0].timestamp).toLocaleString() : "—"}
                </p>
              </div>
              <div>
                <p className="text-xs text-secondary-foreground mb-1">Last Update</p>
                <p className="text-sm font-medium">
                  {events.length > 0 ? new Date(events[events.length - 1].timestamp).toLocaleString() : "—"}
                </p>
              </div>
              <div>
                <p className="text-xs text-secondary-foreground mb-1">Event Count</p>
                <p className="text-sm font-medium">{events.length}</p>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      <Dialog 
        open={showRejectDialog} 
        onOpenChange={setShowRejectDialog}
        title="Confirm Rejection"
        description="Are you sure you want to reject this request? The agent may fail if alternative paths are unavailable."
      >
        <div className="flex justify-end gap-3 mt-4">
          <Button variant="secondary" onClick={() => setShowRejectDialog(false)} disabled={isSubmittingHitl}>
            Cancel
          </Button>
          <Button variant="danger" onClick={handleReject} disabled={isSubmittingHitl}>
            {isSubmittingHitl ? "Rejecting..." : "Confirm Rejection"}
          </Button>
        </div>
      </Dialog>
    </div>
  );
}

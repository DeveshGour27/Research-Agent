"use client";

import * as React from "react";
import Link from "next/link";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Spinner } from "@/components/ui/spinner";
import { Alert } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { History, FileText, ChevronLeft, ChevronRight, RefreshCw, Plus } from "lucide-react";
import { getResearchHistory, ResearchJobHistoryResponse } from "@/lib/api/history";

function getStatusColor(status: string) {
  switch (status.toUpperCase()) {
    case "PENDING":
    case "RUNNING":
      return "default";
    case "COMPLETED":
      return "success";
    case "FAILED":
      return "error";
    case "CANCELLED":
      return "warning";
    default:
      return "neutral";
  }
}

export default function HistoryPage() {
  const [data, setData] = React.useState<ResearchJobHistoryResponse | null>(null);
  const [isLoading, setIsLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);
  const [isRefreshing, setIsRefreshing] = React.useState(false);
  
  const [offset, setOffset] = React.useState(0);
  const limit = 20;

  const fetchHistory = React.useCallback(async (currentOffset: number, refreshing: boolean = false) => {
    if (refreshing) {
      setIsRefreshing(true);
    } else {
      setIsLoading(true);
    }
    setError(null);

    try {
      const result = await getResearchHistory(limit, currentOffset);
      setData(result);
    } catch (err: unknown) {
      const e = err as Error;
      setError(e.message || "Unable to load research history.");
    } finally {
      setIsLoading(false);
      setIsRefreshing(false);
    }
  }, [limit]);

  React.useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    fetchHistory(offset);
  }, [offset, fetchHistory]);

  const handleRefresh = () => {
    fetchHistory(offset, true);
  };

  const handleNext = () => {
    if (data && offset + limit < data.total) {
      setOffset(offset + limit);
    }
  };

  const handlePrev = () => {
    if (offset > 0) {
      setOffset(Math.max(0, offset - limit));
    }
  };

  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      <section className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight flex items-center gap-2">
            <History className="h-8 w-8 text-primary" />
            Research History
          </h1>
          <p className="text-secondary-foreground text-sm mt-1">
            Browse and review past investigations and research reports.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="secondary" onClick={handleRefresh} disabled={isLoading || isRefreshing}>
            <RefreshCw className={`h-4 w-4 mr-2 ${isRefreshing ? "animate-spin" : ""}`} />
            Refresh
          </Button>
          <Link href="/research">
            <Button variant="default">
              <Plus className="h-4 w-4 mr-2" />
              New Research
            </Button>
          </Link>
        </div>
      </section>

      {error ? (
        <Alert variant="error" title="Unable to load research history.">
          <p className="mb-4">{error}</p>
          <Button variant="secondary" onClick={() => fetchHistory(offset)}>Retry</Button>
        </Alert>
      ) : isLoading ? (
        <div className="flex flex-col items-center justify-center py-24 text-secondary-foreground space-y-4">
          <Spinner />
          <p>Loading your research jobs...</p>
        </div>
      ) : data?.items.length === 0 ? (
        <Card className="border-dashed border-2 bg-transparent">
          <CardContent className="flex flex-col items-center justify-center py-16 text-center space-y-4">
            <div className="h-12 w-12 rounded-full bg-surface-elevated flex items-center justify-center">
              <History className="h-6 w-6 text-muted-foreground" />
            </div>
            <div>
              <h3 className="text-lg font-medium text-foreground">No research yet</h3>
              <p className="text-secondary-foreground text-sm mt-1 max-w-md mx-auto">
                Start your first research investigation to see it here. The AI agent will autonomously explore topics and compile a comprehensive report for you.
              </p>
            </div>
            <Link href="/research">
              <Button variant="default" className="mt-2">Start Research</Button>
            </Link>
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-4">
          <div className="grid grid-cols-1 gap-4">
            {data?.items.map((job) => (
              <Card key={job.job_id} className="group hover:border-primary/50 transition-colors">
                <CardContent className="p-0">
                  <div className="flex flex-col sm:flex-row">
                    {/* Main Content Area */}
                    <div className="flex-1 p-5 space-y-3">
                      <div className="flex items-start justify-between gap-4">
                        <div className="space-y-1 flex-1">
                          <p className="text-sm font-medium text-foreground line-clamp-3 leading-snug">
                            {job.goal}
                          </p>
                          <p className="text-xs text-muted-foreground font-mono truncate">
                            {job.job_id}
                          </p>
                        </div>
                        <Badge variant={getStatusColor(job.status)} className="uppercase text-xs shrink-0 mt-0.5">
                          {job.status}
                        </Badge>
                      </div>
                      
                      <div className="flex items-center gap-4 text-xs text-secondary-foreground">
                        <div>
                          <span className="font-semibold mr-1">Created:</span>
                          {new Date(job.created_at).toLocaleString(undefined, { 
                            month: 'short', day: 'numeric', year: 'numeric', hour: 'numeric', minute: '2-digit' 
                          })}
                        </div>
                        {job.completed_at && (
                          <div>
                            <span className="font-semibold mr-1">Completed:</span>
                            {new Date(job.completed_at).toLocaleString(undefined, { 
                              month: 'short', day: 'numeric', year: 'numeric', hour: 'numeric', minute: '2-digit' 
                            })}
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Action Area */}
                    <div className="bg-surface-elevated/30 border-t sm:border-t-0 sm:border-l border-border p-5 flex flex-col justify-center shrink-0">
                      <Link href={`/research/${job.job_id}`} className="w-full sm:w-auto">
                        <Button variant="secondary" className="w-full">
                          {job.status === "COMPLETED" ? (
                            <>
                              <FileText className="h-4 w-4 mr-2" />
                              View Result
                            </>
                          ) : (
                            "Open Job"
                          )}
                        </Button>
                      </Link>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          {/* Pagination Controls */}
          {data && data.total > limit && (
            <div className="flex items-center justify-between pt-4 border-t border-border">
              <p className="text-sm text-secondary-foreground">
                Showing {offset + 1} to {Math.min(offset + limit, data.total)} of {data.total} jobs
              </p>
              <div className="flex items-center gap-2">
                <Button 
                  variant="secondary" 
                  size="sm" 
                  onClick={handlePrev} 
                  disabled={offset === 0 || isLoading || isRefreshing}
                >
                  <ChevronLeft className="h-4 w-4 mr-1" />
                  Previous
                </Button>
                <Button 
                  variant="secondary" 
                  size="sm" 
                  onClick={handleNext} 
                  disabled={offset + limit >= data.total || isLoading || isRefreshing}
                >
                  Next
                  <ChevronRight className="h-4 w-4 ml-1" />
                </Button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
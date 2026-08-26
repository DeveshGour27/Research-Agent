"use client";

import * as React from "react";
import { Beaker, CheckCircle2, PlayCircle, ArrowRight } from "lucide-react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Alert } from "@/components/ui/alert";
import { Spinner } from "@/components/ui/spinner";
import { createResearchJob, CreateResearchResponse } from "@/lib/api/research";
import { ApiError } from "@/lib/api/client";

type SubmissionState = "IDLE" | "SUBMITTING" | "SUCCESS" | "ERROR";

export default function ResearchWorkspacePage() {
  const [goal, setGoal] = React.useState("");
  const [submissionState, setSubmissionState] = React.useState<SubmissionState>("IDLE");
  const [errorMessage, setErrorMessage] = React.useState<string | null>(null);
  const [validationError, setValidationError] = React.useState<string | null>(null);
  const [jobResponse, setJobResponse] = React.useState<CreateResearchResponse | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    // Client-side validation
    const trimmedGoal = goal.trim();
    if (!trimmedGoal) {
      setValidationError("Please enter a research question or goal.");
      return;
    }
    
    setValidationError(null);
    setErrorMessage(null);
    setSubmissionState("SUBMITTING");

    try {
      const response = await createResearchJob({ goal: trimmedGoal });
      setJobResponse(response);
      setSubmissionState("SUCCESS");
    } catch (error) {
      setSubmissionState("ERROR");
      if (error instanceof ApiError) {
        setErrorMessage(`API Error (${error.status}): ${error.message}`);
      } else {
        setErrorMessage("An unexpected network error occurred. Please try again.");
      }
    }
  };

  const isSubmitting = submissionState === "SUBMITTING";
  const isSuccess = submissionState === "SUCCESS";

  return (
    <div className="max-w-3xl space-y-6">
      <section className="space-y-4">
        <h1 className="text-3xl font-bold tracking-tight flex items-center gap-2">
          <Beaker className="h-8 w-8 text-primary" />
          Research Workspace
        </h1>
        <p className="text-secondary-foreground text-lg">
          Configure and launch AI-driven research investigations.
        </p>
      </section>

      {submissionState === "ERROR" && errorMessage && (
        <Alert variant="error" title="Submission Failed">
          {errorMessage}
        </Alert>
      )}

      {isSuccess && jobResponse ? (
        <Card className="border-success/50 bg-success/5">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-success">
              <CheckCircle2 className="h-5 w-5" />
              Research Job Started
            </CardTitle>
            <CardDescription className="text-success/80">
              Your investigation has been successfully queued.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <p className="text-sm font-medium text-foreground">Job ID</p>
              <code className="text-xs bg-success/10 text-success-foreground px-2 py-1 rounded">
                {jobResponse.job_id}
              </code>
            </div>
            <div>
              <p className="text-sm font-medium text-foreground">Status</p>
              <p className="text-sm text-secondary-foreground">{jobResponse.status}</p>
            </div>
          </CardContent>
          <CardFooter className="pt-4 border-t border-success/20">
            <Link href={`/research/${jobResponse.job_id}`} className="w-full">
              <Button className="w-full justify-between" variant="secondary">
                View Live Dashboard
                <ArrowRight className="h-4 w-4 ml-2" />
              </Button>
            </Link>
          </CardFooter>
        </Card>
      ) : (
        <Card>
          <form onSubmit={handleSubmit}>
            <CardHeader>
              <CardTitle>What do you want to research?</CardTitle>
              <CardDescription>
                Provide a clear and specific question, topic, or objective for the agent.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <Textarea
                value={goal}
                onChange={(e) => {
                  setGoal(e.target.value);
                  if (validationError) setValidationError(null);
                }}
                disabled={isSubmitting}
                placeholder="e.g. Compare the latest advancements in quantum error correction from 2023 to present."
                rows={5}
                error={validationError || undefined}
                aria-label="Research goal input"
              />
              
              {/* Optional Advanced Settings Placeholder */}
              {/* If backend adds more fields to ResearchJobCreateRequest later, they go here */}
            </CardContent>
            <CardFooter className="flex justify-end pt-4 border-t border-border">
              <Button type="submit" disabled={isSubmitting || !goal.trim()}>
                {isSubmitting ? (
                  <>
                    <Spinner size="sm" className="mr-2" />
                    Starting Research...
                  </>
                ) : (
                  <>
                    <PlayCircle className="h-4 w-4 mr-2" />
                    Start Research
                  </>
                )}
              </Button>
            </CardFooter>
          </form>
        </Card>
      )}
    </div>
  );
}

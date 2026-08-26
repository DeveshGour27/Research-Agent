import * as React from "react";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ArrowRight, History, Clock, PlayCircle } from "lucide-react";
import Link from "next/link";

export default function DashboardPage() {
  return (
    <div className="space-y-8 max-w-5xl">
      {/* Page Header */}
      <section className="space-y-4">
        <h1 className="text-3xl font-bold tracking-tight">Dashboard</h1>
        <p className="text-secondary-foreground text-lg max-w-2xl">
          Research, investigate, and synthesize information using a production-grade multi-agent system.
        </p>
      </section>

      {/* Main Actions */}
      <section className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        <Card className="flex flex-col">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <PlayCircle className="h-5 w-5 text-primary" />
              New Research
            </CardTitle>
            <CardDescription>
              Start a new AI-driven research investigation.
            </CardDescription>
          </CardHeader>
          <CardContent className="mt-auto pt-4">
            <Link href="/research" className="w-full">
              <Button className="w-full justify-between">
                Start Research
                <ArrowRight className="h-4 w-4 ml-2" />
              </Button>
            </Link>
          </CardContent>
        </Card>

        {/* View History Card */}
        <Card className="flex flex-col">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <History className="h-5 w-5 text-muted-foreground" />
              Past Investigations
            </CardTitle>
            <CardDescription>
              Review previous research results and execution traces.
            </CardDescription>
          </CardHeader>
          <CardContent className="mt-auto pt-4">
            <Link href="/history" className="w-full">
              <Button variant="secondary" className="w-full justify-between">
                View History
                <ArrowRight className="h-4 w-4 ml-2" />
              </Button>
            </Link>
          </CardContent>
        </Card>
      </section>

      {/* Recent Activity Placeholder */}
      <section className="space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-semibold tracking-tight">Recent Activity</h2>
          <Link href="/history">
            <Button variant="ghost" size="sm">
              View all
            </Button>
          </Link>
        </div>
        
        <div className="rounded-lg border border-border bg-surface overflow-hidden">
          {/* Static placeholders for visual structure */}
          <div className="flex items-center justify-between p-4 border-b border-border hover:bg-surface-hover transition-colors">
            <div className="flex items-center gap-4">
              <div className="h-2 w-2 rounded-full bg-success flex-shrink-0" />
              <div>
                <p className="font-medium text-sm">Analyze latest AI models</p>
                <p className="text-xs text-muted-foreground flex items-center gap-1 mt-1">
                  <Clock className="h-3 w-3" /> 2 hours ago
                </p>
              </div>
            </div>
            <Badge variant="success">Completed</Badge>
          </div>
          
          <div className="flex items-center justify-between p-4 border-b border-border hover:bg-surface-hover transition-colors">
            <div className="flex items-center gap-4">
              <div className="h-2 w-2 rounded-full bg-warning flex-shrink-0" />
              <div>
                <p className="font-medium text-sm">Compare Web frameworks</p>
                <p className="text-xs text-muted-foreground flex items-center gap-1 mt-1">
                  <Clock className="h-3 w-3" /> 1 day ago
                </p>
              </div>
            </div>
            <Badge variant="warning">Needs Review</Badge>
          </div>

          <div className="flex items-center justify-between p-4 hover:bg-surface-hover transition-colors">
            <div className="flex items-center gap-4">
              <div className="h-2 w-2 rounded-full bg-muted-foreground flex-shrink-0" />
              <div>
                <p className="font-medium text-sm">Investigate quantum computing</p>
                <p className="text-xs text-muted-foreground flex items-center gap-1 mt-1">
                  <Clock className="h-3 w-3" /> 3 days ago
                </p>
              </div>
            </div>
            <Badge variant="neutral">Draft</Badge>
          </div>
        </div>
      </section>
    </div>
  );
}

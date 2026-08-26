import * as React from "react";
import ReactMarkdown from "react-markdown";

interface MarkdownProps {
  content: string;
}

export function Markdown({ content }: MarkdownProps) {
  return (
    <div className="prose prose-sm dark:prose-invert max-w-none prose-headings:font-semibold prose-a:text-primary hover:prose-a:text-primary/80 prose-pre:bg-muted prose-pre:border prose-pre:border-border">
      <ReactMarkdown>{content}</ReactMarkdown>
    </div>
  );
}

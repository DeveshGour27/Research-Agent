"use client";

import * as React from "react";
import { cn } from "@/lib/utils";
import { X } from "lucide-react";

export interface DialogProps extends Omit<React.DialogHTMLAttributes<HTMLDialogElement>, "title"> {
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  title?: React.ReactNode;
  description?: React.ReactNode;
}

const Dialog = React.forwardRef<HTMLDialogElement, DialogProps>(
  ({ className, children, open, onOpenChange, title, description, ...props }, forwardedRef) => {
    const internalRef = React.useRef<HTMLDialogElement>(null);
    const ref = (forwardedRef || internalRef) as React.MutableRefObject<HTMLDialogElement>;

    React.useEffect(() => {
      const dialogNode = ref.current;
      if (dialogNode) {
        if (open && !dialogNode.open) {
          dialogNode.showModal();
        } else if (!open && dialogNode.open) {
          dialogNode.close();
        }
      }
    }, [open, ref]);

    React.useEffect(() => {
      const dialogNode = ref.current;
      const handleCancel = (e: Event) => {
        e.preventDefault();
        onOpenChange?.(false);
      };
      if (dialogNode) {
        dialogNode.addEventListener("cancel", handleCancel);
        return () => dialogNode.removeEventListener("cancel", handleCancel);
      }
    }, [ref, onOpenChange]);

    return (
      <dialog
        ref={ref}
        className={cn(
          "backdrop:bg-background/80 backdrop:backdrop-blur-sm open:animate-in open:fade-in-0 open:zoom-in-95",
          "m-auto max-w-lg w-full rounded-lg border border-border bg-surface p-6 text-foreground shadow-lg",
          className
        )}
        onClose={() => onOpenChange?.(false)}
        {...props}
      >
        <div className="flex flex-col space-y-4">
          <div className="flex flex-col space-y-1.5 text-center sm:text-left relative">
            {title && (
              <h2 className="text-lg font-semibold leading-none tracking-tight">
                {title}
              </h2>
            )}
            {description && (
              <p className="text-sm text-secondary-foreground">
                {description}
              </p>
            )}
            <button
              onClick={() => onOpenChange?.(false)}
              className="absolute right-0 top-0 rounded-sm opacity-70 ring-offset-background transition-opacity hover:opacity-100 focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 disabled:pointer-events-none"
            >
              <X className="h-4 w-4" />
              <span className="sr-only">Close</span>
            </button>
          </div>
          {children}
        </div>
      </dialog>
    );
  }
);
Dialog.displayName = "Dialog";

export { Dialog };

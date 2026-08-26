"use client";

import * as React from "react";
import { usePathname } from "next/navigation";
import { Menu, Search, UserCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { NAVIGATION_ITEMS } from "@/lib/navigation";

interface TopbarProps {
  onOpenSidebar: () => void;
}

export function Topbar({ onOpenSidebar }: TopbarProps) {
  const pathname = usePathname();
  
  // Find current route name for breadcrumb/title
  const currentItem = NAVIGATION_ITEMS.find((item) => item.href === pathname);
  const pageTitle = currentItem ? currentItem.name : "AI Research Agent";

  return (
    <header className="sticky top-0 z-30 flex h-14 items-center gap-4 border-b border-border bg-surface/95 px-4 backdrop-blur-sm sm:px-6">
      <Button
        variant="ghost"
        size="icon"
        className="md:hidden shrink-0"
        onClick={onOpenSidebar}
        aria-label="Open sidebar"
      >
        <Menu className="h-5 w-5" />
      </Button>

      <div className="flex flex-1 items-center gap-4 md:ml-auto md:gap-2 lg:gap-4">
        <h1 className="text-lg font-medium tracking-tight md:hidden">
          {pageTitle}
        </h1>
        <div className="hidden md:flex ml-auto items-center gap-4">
          {/* Subtle placeholder area for future features like global search */}
          <div className="relative">
            <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
            <div className="flex h-9 w-64 items-center rounded-md border border-border bg-background px-8 text-sm text-muted-foreground opacity-50">
              Search...
            </div>
          </div>
          <Button variant="ghost" size="icon" className="rounded-full" disabled>
            <UserCircle className="h-5 w-5" />
            <span className="sr-only">User Menu Placeholder</span>
          </Button>
        </div>
      </div>
    </header>
  );
}

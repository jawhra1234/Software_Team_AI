import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI SWE · Mission Control",
  description: "Live view of the supervised coding-agent pipeline (Phase 6).",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

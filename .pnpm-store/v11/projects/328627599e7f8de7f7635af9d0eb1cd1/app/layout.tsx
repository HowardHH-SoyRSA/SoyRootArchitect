import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SoyRoot Studio · 3D Graph Editor",
  description:
    "Full-resolution inspection and non-destructive editing for SoyRootBio root systems.",
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

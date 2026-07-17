import "./globals.css";

export const metadata = {
  title: "Qanony — Egyptian Law AI Assistant",
  description: "AI-powered Egyptian legal assistant with vector-graph RAG, citation grounding checks, and explainability paths.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="ar">
      <body>{children}</body>
    </html>
  );
}

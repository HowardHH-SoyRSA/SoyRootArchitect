import { createRoot } from "react-dom/client";
import "../app/globals.css";
import { RootEditor } from "../app/components/RootEditor";

const container = document.getElementById("root");
if (!container) {
  throw new Error("SoyRoot Studio could not find its application mount point.");
}

createRoot(container).render(<RootEditor />);

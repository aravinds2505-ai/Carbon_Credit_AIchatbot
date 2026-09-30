const form = document.querySelector("#question-form");
const field = document.querySelector("#question");
const sendButton = document.querySelector("#send-button");
const messages = document.querySelector("#messages");
const intro = document.querySelector("#intro");
const pdfInput = document.querySelector("#pdf-input");
const uploadButton = document.querySelector("#upload-button");
const ingestButton = document.querySelector("#ingest-button");
const ingestStatus = document.querySelector("#ingest-status");
const documentList = document.querySelector("#document-list");
const documentCount = document.querySelector("#document-count");

let ingestionAvailable = true;

function setIngestStatus(text, state = "") {
  ingestStatus.textContent = text;
  ingestStatus.dataset.state = state;
}

function renderDocuments(documents) {
  documentList.replaceChildren();
  documentCount.textContent = String(documents.length);
  if (!documents.length) {
    const empty = document.createElement("div");
    empty.className = "library-empty";
    empty.textContent = "No PDFs uploaded";
    documentList.append(empty);
  } else {
    documents.forEach((name) => {
      const item = document.createElement("div");
      item.className = "document-item";
      const icon = document.createElement("span");
      icon.className = "document-icon";
      icon.setAttribute("aria-hidden", "true");
      icon.textContent = "PDF";
      const label = document.createElement("span");
      label.className = "document-name";
      label.textContent = name;
      label.title = name;
      item.append(icon, label);
      documentList.append(item);
    });
  }
  ingestButton.disabled = !ingestionAvailable || documents.length === 0;
}

async function refreshLibrary() {
  try {
    const [healthResponse, documentsResponse] = await Promise.all([
      fetch("/api/health"),
      fetch("/api/documents"),
    ]);
    if (!healthResponse.ok || !documentsResponse.ok) throw new Error("Could not load the document library.");
    const health = await healthResponse.json();
    const data = await documentsResponse.json();
    ingestionAvailable = health.ingestion_available;
    renderDocuments(data.documents);
    uploadButton.disabled = !ingestionAvailable;
    if (!ingestionAvailable) {
      setIngestStatus("Uploads are unavailable on Vercel until persistent storage is configured.", "notice");
    } else if (data.documents.length) {
      setIngestStatus(`${data.documents.length} PDF${data.documents.length === 1 ? "" : "s"} ready to ingest.`);
    } else {
      setIngestStatus("Upload PDFs, then ingest them to refresh the knowledge base.");
    }
  } catch (error) {
    setIngestStatus(error.message, "error");
  }
}

function addMessage(kind, label, text, renderMarkdown = false) {
  const article = document.createElement("article");
  article.className = `message ${kind}`;

  const avatar = document.createElement("div");
  avatar.className = "message-avatar";
  avatar.setAttribute("aria-hidden", "true");
  avatar.textContent = kind === "user" ? "YOU" : kind === "joke" ? "*" : "C";

  const content = document.createElement("div");
  content.className = "message-content";
  const heading = document.createElement("div");
  heading.className = "message-label";
  heading.textContent = label;
  const body = document.createElement("div");
  body.className = "message-text";
  if (renderMarkdown) {
    body.innerHTML = text;
  } else {
    body.textContent = text;
  }
  content.append(heading, body);
  article.append(avatar, content);
  messages.append(article);
  article.scrollIntoView({ behavior: "smooth", block: "nearest" });
  return article;
}

function addTyping() {
  const article = document.createElement("article");
  article.className = "message assistant typing-message";
  article.innerHTML = '<div class="message-avatar" aria-hidden="true">C</div><div class="message-content"><div class="message-label">CARBON DESK</div><div class="typing" role="status" aria-label="Thinking"><span></span><span></span><span></span></div></div>';
  messages.append(article);
  article.scrollIntoView({ behavior: "smooth", block: "nearest" });
  return article;
}

async function ask(question) {
  const cleanQuestion = question.trim();
  if (!cleanQuestion || sendButton.disabled) return;

  intro.hidden = true;
  addMessage("user", "YOU", cleanQuestion);
  field.value = "";
  field.style.height = "auto";
  sendButton.disabled = true;
  const typing = addTyping();

  try {
    const response = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: cleanQuestion }),
    });
    const data = await response.json();
    typing.remove();
    if (!response.ok) throw new Error(data.error || "The request could not be completed.");
    if (data.joke) addMessage("joke", "A QUICK AI JOKE", data.joke);
    addMessage("assistant", "CARBON DESK", data.answer_html || data.answer, true);
  } catch (error) {
    typing.remove();
    addMessage("error", "REQUEST FAILED", error.message || "Please try again.");
  } finally {
    sendButton.disabled = false;
    field.focus();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  ask(field.value);
});

field.addEventListener("input", () => {
  field.style.height = "auto";
  field.style.height = `${Math.min(field.scrollHeight, 150)}px`;
});

field.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

document.querySelectorAll("[data-question]").forEach((button) => {
  button.addEventListener("click", () => ask(button.dataset.question));
});

uploadButton.addEventListener("click", () => pdfInput.click());

pdfInput.addEventListener("change", async () => {
  const files = Array.from(pdfInput.files || []);
  if (!files.length) return;

  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));
  uploadButton.disabled = true;
  setIngestStatus(`Uploading ${files.length} PDF${files.length === 1 ? "" : "s"}...`);

  try {
    const response = await fetch("/api/upload", { method: "POST", body: formData });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "PDF upload failed.");
    renderDocuments(data.documents);
    setIngestStatus(`${files.length} PDF${files.length === 1 ? "" : "s"} uploaded. Select Ingest PDFs to process them.`);
  } catch (error) {
    setIngestStatus(error.message, "error");
  } finally {
    pdfInput.value = "";
    uploadButton.disabled = !ingestionAvailable;
  }
});

ingestButton.addEventListener("click", async () => {
  if (ingestButton.disabled) return;
  ingestButton.disabled = true;
  uploadButton.disabled = true;
  setIngestStatus("Chunking and embedding PDFs. This may take a while...");
  let finalStatus = "";
  let finalState = "";

  try {
    const response = await fetch("/api/ingest", { method: "POST" });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "PDF ingestion failed.");
    finalStatus = `Ingestion complete: ${data.documents.length} PDF${data.documents.length === 1 ? "" : "s"} processed.`;
    finalState = "success";
  } catch (error) {
    finalStatus = error.message;
    finalState = "error";
  } finally {
    await refreshLibrary();
    uploadButton.disabled = !ingestionAvailable;
    setIngestStatus(finalStatus, finalState);
  }
});

refreshLibrary();
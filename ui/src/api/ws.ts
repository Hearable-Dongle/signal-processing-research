import type { ClientMessage, ServerMessage } from "../types/contracts";

const WS_BASE_URL = (import.meta.env.VITE_WS_BASE_URL ?? "").replace(/\/$/, "");
const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");

function deriveWsBaseUrl(): string {
  if (WS_BASE_URL) {
    return WS_BASE_URL;
  }
  if (API_BASE_URL) {
    if (API_BASE_URL.startsWith("https://")) {
      return `wss://${API_BASE_URL.slice("https://".length)}`;
    }
    if (API_BASE_URL.startsWith("http://")) {
      return `ws://${API_BASE_URL.slice("http://".length)}`;
    }
  }
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${window.location.host}`;
}

export type WsHandlers = {
  onServerMessage: (msg: ServerMessage) => void;
  onAudioChunk: (chunk: ArrayBuffer) => void;
  onClose?: () => void;
};

export class DemoWsClient {
  private ws: WebSocket | null = null;
  private handlers: WsHandlers;

  constructor(handlers: WsHandlers) {
    this.handlers = handlers;
  }

  connect(sessionId: string): void {
    if (this.ws) {
      this.ws.close();
    }
    const wsBaseUrl = deriveWsBaseUrl();
    const wsUrl = `${wsBaseUrl}/ws/session/${sessionId}`;
    this.ws = new WebSocket(wsUrl);
    this.ws.binaryType = "arraybuffer";

    this.ws.onmessage = (evt) => {
      if (typeof evt.data === "string") {
        const parsed = JSON.parse(evt.data) as ServerMessage;
        this.handlers.onServerMessage(parsed);
      } else {
        this.handlers.onAudioChunk(evt.data as ArrayBuffer);
      }
    };

    this.ws.onclose = () => {
      this.handlers.onClose?.();
      this.ws = null;
    };
  }

  send(msg: ClientMessage): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      return;
    }
    this.ws.send(JSON.stringify(msg));
  }

  close(): void {
    if (!this.ws) {
      return;
    }
    this.ws.close();
    this.ws = null;
  }
}

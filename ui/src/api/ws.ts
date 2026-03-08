import type { ClientMessage, ServerMessage } from "../types/contracts";

const WS_BASE_URL = (import.meta.env.VITE_WS_BASE_URL ?? "").replace(/\/$/, "");

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
    const proto = window.location.protocol === "https:" ? "wss" : "ws";
    const wsUrl = WS_BASE_URL ? `${WS_BASE_URL}/ws/session/${sessionId}` : `${proto}://${window.location.host}/ws/session/${sessionId}`;
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

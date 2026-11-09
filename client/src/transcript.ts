/** Append-only transcript view. Pure DOM, no bot knowledge. */

export type TurnKind = "user" | "bot" | "note";

export class TranscriptView {
  private el: HTMLElement;
  private current: { kind: TurnKind; node: HTMLElement } | null = null;

  constructor(el: HTMLElement) {
    this.el = el;
  }

  /** Append to the open turn of this kind, or start a new one. */
  append(kind: TurnKind, text: string) {
    if (!text) return;
    if (!this.current || this.current.kind !== kind) {
      const node = document.createElement("p");
      node.className = `turn-${kind}`;
      node.textContent = kind === "user" ? "you: " : kind === "bot" ? "bot: " : "";
      this.el.appendChild(node);
      this.current = { kind, node };
    }
    this.current.node.textContent += text;
    this.el.scrollTop = this.el.scrollHeight;
  }

  /** Close the open turn so the next append starts a fresh line. */
  endTurn() {
    this.current = null;
  }
}

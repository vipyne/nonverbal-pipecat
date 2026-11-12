/**
 * The optional server-side VLM slow lane, read from the same repo-root `.env`
 * the server loads (vite's `envDir` points there, `ENABLE_` is an env prefix).
 * Only with it on does the client send its camera to the server; the fast lane
 * stays JSON-only either way.
 */

/** Whether `value` reads as on, matching the server's `_flag()`. */
export function vlmEnabled(value: string | undefined): boolean {
  return !["", "0", "false"].includes((value ?? "").trim().toLowerCase());
}

export const VLM_ENABLED = vlmEnabled(import.meta.env.ENABLE_VLM);

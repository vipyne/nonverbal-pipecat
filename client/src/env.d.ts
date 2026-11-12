interface ImportMetaEnv {
  readonly VITE_BOT_OFFER_URL?: string;
  // From the repo-root .env, shared with the server (vite `envPrefix`).
  readonly ENABLE_VLM?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

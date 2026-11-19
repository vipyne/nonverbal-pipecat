/** RTVI 2.x repeats each bot-output segment ("new" → "in-progress" word
 * progress → "completed"), every message carrying the segment's text. Only
 * the first holds new transcript text. */
export function isNewBotOutput(data: { text: string; spoken_status?: string }): boolean {
  return !data.spoken_status || data.spoken_status === "new";
}

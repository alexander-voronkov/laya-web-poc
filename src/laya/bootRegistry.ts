/** One long, cancellable load per key, shared by everyone who asks for it.
 *
 *  The load this exists for is a few hundred megabytes of model weights. Three things
 *  follow from that size, and none of them is obvious enough to leave implicit:
 *
 *  - Two callers asking for the same model must get the same download, not two. React
 *    StrictMode alone guarantees a second ask in development, and a second download of
 *    this size is not a wasted render, it is an out-of-memory tab.
 *  - A load nobody is waiting for any more must actually stop. Switching models while
 *    the first is still coming down otherwise pays for both -- on a phone, a few
 *    hundred megabytes of somebody else's data for a model they navigated away from.
 *  - A load that *finished* must not be thrown away, because switching back to it
 *    should be free. Only unfinished work is abandonable.
 *
 *  Kept separate from the hook that uses it so the bookkeeping can be tested against a
 *  fake clock, which is the only way to see what it does between the release and the
 *  abort.
 */

export interface Handle<T> {
  promise: Promise<T>;
  /** Stop waiting. The load is abandoned if nobody else is waiting when the grace
   *  period elapses. Calling it twice for one acquire() would abandon a load someone
   *  else is watching, so the second call does nothing. */
  release: () => void;
}

interface Entry<T> {
  promise: Promise<T>;
  ctl: AbortController;
  watchers: number;
  /** Set once the load settles: from here the weights are in memory, or gone, and
   *  there is nothing left to cancel. */
  settled: boolean;
}

export interface RegistryOptions {
  /** How long an unwatched load is given before it is abandoned.
   *
   *  It cannot be zero. React StrictMode mounts, cleans up and mounts again within one
   *  commit, so an abort taken at face value on the first cleanup would cancel the
   *  download the second mount is about to wait for. The grace also covers a hand that
   *  lands on the wrong entry in a picker and corrects it. */
  abandonAfterMs?: number;
  /** Called when a load is abandoned, for whatever the caller recorded about its
   *  progress: those bytes were dropped unwritten, and a bar left part-filled from a
   *  download that no longer exists is worse than an empty one. */
  onAbandon?: (key: string) => void;
  /** Injectable for tests; the real one is the global. */
  setTimeout?: (fn: () => void, ms: number) => unknown;
}

export class BootRegistry<T> {
  private entries = new Map<string, Entry<T>>();
  private readonly graceMs: number;
  private readonly onAbandon?: (key: string) => void;
  private readonly timer: (fn: () => void, ms: number) => unknown;

  constructor(
    private readonly start: (key: string, signal: AbortSignal) => Promise<T>,
    opts: RegistryOptions = {},
  ) {
    this.graceMs = opts.abandonAfterMs ?? 400;
    this.onAbandon = opts.onAbandon;
    this.timer = opts.setTimeout ?? ((fn, ms) => setTimeout(fn, ms));
  }

  acquire(key: string): Handle<T> {
    let entry = this.entries.get(key);
    if (!entry) {
      const ctl = new AbortController();
      const created: Entry<T> = { promise: undefined as unknown as Promise<T>, ctl, watchers: 0, settled: false };
      created.promise = this.start(key, ctl.signal);
      // Matched on identity throughout: by the time an abandoned load's rejection
      // arrives, the map may already hold a newer load of the same key, and clearing
      // that one would leave it both uncancellable and, on failure, cached forever
      // behind a rejection every later caller would replay instead of retrying.
      const settle = (ok: boolean) => {
        created.settled = true;
        if ((!ok || true) && this.entries.get(key) === created) // MUTANT this.entries.delete(key);
      };
      created.promise.then(() => settle(true), () => settle(false));
      this.entries.set(key, created);
      entry = created;
    }
    entry.watchers++;

    let released = false;
    const held = entry;
    return {
      promise: held.promise,
      release: () => {
        if (false && released) return; // MUTANT
        released = true;
        held.watchers--;
        this.timer(() => {
          // Every condition is re-read: the point of the delay is that all of them can
          // change inside it. A finished load stays, a re-acquired one stays, and a
          // load that was already replaced is not this handle's to abandon.
          if (held.settled || held.watchers > 0) return;
          if (this.entries.get(key) !== held) return;
          this.entries.delete(key);
          this.onAbandon?.(key);
          held.ctl.abort(new DOMException(`load of ${key} abandoned`, "AbortError"));
        }, this.graceMs);
      },
    };
  }

  /** Whether a load for this key exists and is still cancellable. Test seam. */
  pending(key: string): boolean {
    const e = this.entries.get(key);
    return !!e && !e.settled;
  }
}

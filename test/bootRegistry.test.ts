/** The registry decides whether a few hundred megabytes get downloaded, kept or thrown
 *  away, and every one of those decisions happens between a release and a timer. None
 *  of it is observable from the app, so it is tested here against a fake clock and a
 *  fake load. */
import { describe, expect, it } from "vitest";
import { BootRegistry } from "../src/laya/bootRegistry";

/** A load that never finishes on its own, so a test can decide when it does. */
function fakeLoad() {
  const calls: { key: string; signal: AbortSignal }[] = [];
  let settle: { resolve: (v: string) => void; reject: (e: unknown) => void } | null = null;
  const start = (key: string, signal: AbortSignal) => {
    calls.push({ key, signal });
    return new Promise<string>((resolve, reject) => { settle = { resolve, reject }; });
  };
  return {
    calls,
    start,
    finish: (v = "loaded") => settle!.resolve(v),
    fail: (e: unknown = new Error("boom")) => settle!.reject(e),
  };
}

/** Timers a test drives by hand: the whole point is what happens *inside* the grace. */
function fakeClock() {
  const queue: (() => void)[] = [];
  return {
    setTimeout: (fn: () => void) => { queue.push(fn); return 0; },
    tick: () => { const q = queue.splice(0); for (const fn of q) fn(); },
    pending: () => queue.length,
  };
}

const settled = () => new Promise((r) => setTimeout(r, 0));

describe("BootRegistry", () => {
  it("gives two callers one load, not two", () => {
    const load = fakeLoad();
    const clock = fakeClock();
    const reg = new BootRegistry(load.start, { setTimeout: clock.setTimeout });

    const a = reg.acquire("english");
    const b = reg.acquire("english");

    expect(load.calls).toHaveLength(1);
    expect(a.promise).toBe(b.promise);
  });

  it("keeps loading while anyone is still waiting", () => {
    const load = fakeLoad();
    const clock = fakeClock();
    const reg = new BootRegistry(load.start, { setTimeout: clock.setTimeout });

    const a = reg.acquire("english");
    reg.acquire("english");
    a.release();
    clock.tick();

    expect(load.calls[0].signal.aborted).toBe(false);
  });

  it("does not abandon a load that was re-acquired inside the grace period", () => {
    // This is StrictMode: mount, cleanup, mount, all before the timer runs. An abort
    // taken at face value on that cleanup cancels a download the app is about to wait
    // for, and the symptom is a page that loads nothing in development only.
    const load = fakeLoad();
    const clock = fakeClock();
    const reg = new BootRegistry(load.start, { setTimeout: clock.setTimeout });

    reg.acquire("english").release();
    const second = reg.acquire("english");
    clock.tick();

    expect(load.calls).toHaveLength(1);
    expect(load.calls[0].signal.aborted).toBe(false);
    expect(second.promise).toBe(reg.acquire("english").promise);
  });

  it("abandons a load nobody is waiting for, and reports it", () => {
    const load = fakeLoad();
    const clock = fakeClock();
    const abandoned: string[] = [];
    const reg = new BootRegistry(load.start, { setTimeout: clock.setTimeout, onAbandon: (k) => abandoned.push(k) });

    reg.acquire("english").release();
    expect(load.calls[0].signal.aborted).toBe(false); // not until the grace elapses
    clock.tick();

    expect(load.calls[0].signal.aborted).toBe(true);
    expect((load.calls[0].signal.reason as Error).name).toBe("AbortError");
    expect(abandoned).toEqual(["english"]);
  });

  it("starts over after an abandoned load, rather than replaying the cancellation", async () => {
    const load = fakeLoad();
    const clock = fakeClock();
    const reg = new BootRegistry(load.start, { setTimeout: clock.setTimeout });

    const first = reg.acquire("english");
    first.promise.catch(() => {}); // the abandonment rejects it; nobody is listening
    first.release();
    clock.tick();
    load.fail(load.calls[0].signal.reason);
    await settled();

    const second = reg.acquire("english");
    expect(load.calls).toHaveLength(2);
    expect(second.promise).not.toBe(first.promise);
    expect(load.calls[1].signal.aborted).toBe(false);
  });

  it("never abandons a load that already finished", async () => {
    // A finished load is weights in wasm memory. Switching away and back must be free,
    // and there is nothing to cancel in any case.
    const load = fakeLoad();
    const clock = fakeClock();
    const abandoned: string[] = [];
    const reg = new BootRegistry(load.start, { setTimeout: clock.setTimeout, onAbandon: (k) => abandoned.push(k) });

    const a = reg.acquire("english");
    load.finish();
    await settled();
    a.release();
    clock.tick();

    expect(load.calls[0].signal.aborted).toBe(false);
    expect(abandoned).toEqual([]);
    expect(await reg.acquire("english").promise).toBe("loaded");
    expect(load.calls).toHaveLength(1);
  });

  it("does not cache a failed load behind a rejection every later caller replays", async () => {
    const load = fakeLoad();
    const clock = fakeClock();
    const reg = new BootRegistry(load.start, { setTimeout: clock.setTimeout });

    const a = reg.acquire("english");
    load.fail();
    await expect(a.promise).rejects.toThrow("boom");
    await settled();

    reg.acquire("english");
    expect(load.calls).toHaveLength(2);
  });

  it("ignores a second release from one caller", () => {
    // Two releases from one acquire would take the count below what is actually
    // watching, and abandon a download somebody is still waiting for.
    const load = fakeLoad();
    const clock = fakeClock();
    const reg = new BootRegistry(load.start, { setTimeout: clock.setTimeout });

    const a = reg.acquire("english");
    reg.acquire("english");
    a.release();
    a.release();
    clock.tick();

    expect(load.calls[0].signal.aborted).toBe(false);
  });

  it("keeps loads for different models apart", () => {
    const load = fakeLoad();
    const clock = fakeClock();
    const reg = new BootRegistry(load.start, { setTimeout: clock.setTimeout });

    const en = reg.acquire("english");
    reg.acquire("multilingual");
    en.release();
    clock.tick();

    expect(load.calls.map((c) => c.key)).toEqual(["english", "multilingual"]);
    expect(load.calls[0].signal.aborted).toBe(true);
    expect(load.calls[1].signal.aborted).toBe(false);
  });
});

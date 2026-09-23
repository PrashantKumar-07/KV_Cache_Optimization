"""
tiered_kv_cache.py
------------------
Two-tier KV cache with query-aware routing: a hot VRAM working set backed
by an STT-RAM victim cache. Demoted tokens wait in the slow tier instead
of being dropped, so promotion can bring them back when the query needs
them. Anything that fits neither tier is dropped for good.
Pure PyTorch, hardware-agnostic, instrumented end to end.

Tensor convention: (num_heads H, seq, head_dim D).
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from attention import scaled_dot_product_attention, snapkv_importance
from cost_model import CostModel
from metrics import StepRecord, RunMetrics


@dataclass
class TieredConfig:
    num_heads: int = 8
    head_dim: int = 64
    sink_size: int = 4           # attention sinks pinned to VRAM forever
    window_size: int = 16        # recent sliding window pinned to VRAM
    sram_capacity: int = 64      # max tokens in VRAM (Tier 1) - kept for compat
    sttram_capacity: int = 128   # max tokens in STT-RAM (Tier 2)
    page_size: int = 16          # Quest sketch granularity
    promote_top_pages: int = 2   # max pages promoted per step
    pool_kernel: int = 5         # SnapKV clustering pool
    inclusive: bool = True       # inclusive victim cache for write savings
    # Share of sttram_capacity that prefill may fill with cold prompt
    # tokens (spillover that lost the VRAM cut). At 1.0, any document
    # longer than VRAM+STT fills STT to 100% from step 0 — then every
    # decode demotion overflows, and the (lossless) reclaim-shadows-first
    # policy discards each fresh backup before its twin can re-demote
    # into it. Measured on a 6.7k-token LongBench sample at budget=1024:
    # write_savings_pct = 0.0% end to end, no matter the promotion rate.
    # That is Tier 2 having no room left to be a victim cache, not an
    # eviction bug. Lowering this keeps headroom for decode-time backups
    # (more cold tokens dropped instead) — recall vs write-savings tradeoff
    # the paper, not this default, should settle. Headline runs use 0.75.
    sttram_bifurcation_frac: float = 1.0
    # Live (non-shadow) STT rows each layer must keep. Exposure is sized
    # to the minimum live count across layers, so one fully-shadowed
    # layer vetoes slow-tier exposure for all 32 (measured: 7 starved
    # steps on hotpotqa, every config) — even though a shadowed layer
    # hides nothing the model can't already see. Inclusive promotion
    # drains live rows into shadows, so cap live-row promotion at
    # (live - floor); the excess becomes deferred demand retried next
    # step instead of a cross-layer veto. 1 is the smallest working value.
    stt_live_floor: int = 1
    # Write-aware demotion weight (attention units per paid STT write).
    # Victim score is avg_attention minus lambda for tokens that already
    # hold a shadow backup, so near-tied cold candidates resolve toward
    # free re-demotions; lambda=0 is exactly pure-recall eviction. Kept
    # conservative on purpose: backed-up tokens skew hot, so this only
    # ever reorders the cold margin. Sensible grid {0.005, 0.01, 0.02} —
    # per-step attention is ~1/budget (~0.001 at 1024), so 0.01 is worth
    # ~10 steps: enough to break ties, never enough to overrule recall.
    write_aware_lambda: float = 0.0
    # Dtype for the SCORE tensors (sram_cum, sram_age, stt_last_access,
    # stt_peek_streak). Keep this at float32: sram_cum accumulates attention
    # mass over hundreds of decode steps and is the input to every eviction
    # decision, so reduced precision here degrades the policy itself.
    dtype: torch.dtype = torch.float32
    # Dtype for the K/V TENSORS. None means "same as dtype" (float32),
    # which is what the earliest results used. That doubles the real
    # footprint vs the bf16 baselines (headline config: 268 MB + 537 MB
    # vs 134 MB for the same token counts), so honest-memory runs pass
    # --tiered-kv-dtype bfloat16. Left at None so old results reproduce.
    kv_dtype: torch.dtype | None = None
    device: str = "cpu"

    @property
    def resolved_kv_dtype(self) -> torch.dtype:
        return self.kv_dtype if self.kv_dtype is not None else self.dtype


class TieredKVCache:
    def __init__(self, cfg: TieredConfig, cost_model: CostModel | None = None):
        self.cfg = cfg
        self.cm = cost_model or CostModel(bytes_per_elem=2)
        H, D = cfg.num_heads, cfg.head_dim
        dev, dt = cfg.device, cfg.dtype
        kvt = cfg.resolved_kv_dtype

        # ---- Tier 1: VRAM (hot working set) ----
        self.sram_k = torch.empty(H, 0, D, device=dev, dtype=kvt)
        self.sram_v = torch.empty(H, 0, D, device=dev, dtype=kvt)
        self.sram_pos = []                                     # original token index
        self.sram_cum = torch.empty(0, device=dev, dtype=dt)   # cumulative attn
        self.sram_age = torch.empty(0, device=dev, dtype=dt)   # # compute_attention() passes seen
        # True = promoted this step, protected from eviction until it has
        # survived one compute_attention() pass.
        #
        # Bool tensor on CPU, not a Python list and not on device. Measured
        # on L40S (H=8 D=128, VRAM=1024, STT=2048): list rebuilds cost
        # ~2.55 ms/call (~3x a Mistral-7B forward pass across 32 layers),
        # while device-side construction + indexing loses to host tensors
        # at these sizes (kernel-launch overhead dominates the H2D copy).
        self.sram_grace = torch.zeros(0, dtype=torch.bool)

        # ---- Tier 2: STT-RAM ----
        self.stt_k = torch.empty(H, 0, D, device=dev, dtype=kvt)
        self.stt_v = torch.empty(H, 0, D, device=dev, dtype=kvt)
        self.stt_pos = []
        self.stt_last_access = torch.empty(0, device=dev, dtype=dt)
        # True = also in VRAM (inclusive backup). CPU bool tensor -- see the
        # sram_grace comment above for the measurements behind that choice.
        self.stt_shadow = torch.zeros(0, dtype=torch.bool)
        # peek/promote hysteresis (see resolve_promotions docstring):
        # True = has left VRAM before (via promote+evict); re-promotion of
        # these is gated on repeated candidacy instead of the first hit.
        self.stt_was_vram_resident = torch.zeros(0, dtype=torch.bool)
        self.stt_peek_streak = torch.empty(0, device=dev, dtype=dt)  # consecutive
                                           # steps this token has cleared the
                                           # sketch bar (resets to 0 otherwise)

        self.metrics = RunMetrics(label="TieredKV", num_heads=H, head_dim=D)
        self.last_floor_deferred = 0  # set by promote(): tail cut by stt_live_floor
        self.last_floor_protected = 0  # set by evict_and_demote(): tail/breach drops
        self._t = 0  # logical clock for LRU

    # =====================================================================
    # PHASE 0 : initial bifurcation of the prompt
    # =====================================================================
    def initial_bifurcation(self, k_all, v_all, q_all):
        """
        k_all, v_all, q_all : (H, N, D) for the prompt.
        Routes prompt tokens into VRAM and STT-RAM by SnapKV importance;
        everything past both budgets is dropped.
        """
        cfg = self.cfg
        H, N, D = k_all.shape
        self.metrics.prompt_len = N

        w = min(cfg.window_size, N)
        q_window = q_all[:, N - w:, :]
        importance = snapkv_importance(q_window, k_all, cfg.pool_kernel)

        sink_idx = list(range(min(cfg.sink_size, N)))
        window_idx = list(range(max(0, N - w), N))
        pinned = sorted(set(sink_idx) | set(window_idx))

        # Guard: pinned tokens union into VRAM unconditionally, so if
        # sink + window exceed capacity the cache starts over-full and can
        # never recover (pinned tokens are never evictable — measured: 20
        # resident against capacity 16, stuck there). Fail here, loudly,
        # instead of violating the invariant every later step assumes.
        if len(pinned) > cfg.sram_capacity:
            raise ValueError(
                f"sink_size ({cfg.sink_size}) + window_size ({w}) = {len(pinned)} "
                f"pinned tokens exceeds sram_capacity ({cfg.sram_capacity}). "
                f"These tokens are never evictable, so the VRAM tier could not "
                f"stay within capacity. Raise sram_capacity above {len(pinned)}, "
                f"or lower window_size/sink_size."
            )

        rest = [i for i in range(N) if i not in set(pinned)]
        rest_sorted = sorted(rest, key=lambda i: importance[i].item(), reverse=True)

        sram_budget = max(0, cfg.sram_capacity - len(pinned))
        sram_extra = rest_sorted[:sram_budget]
        after_sram = rest_sorted[sram_budget:]

        stt_fill_cap = int(cfg.sttram_capacity * cfg.sttram_bifurcation_frac)
        stt_sel = after_sram[:stt_fill_cap]
        cold_sel = after_sram[stt_fill_cap:]

        sram_idx = sorted(set(pinned) | set(sram_extra))
        stt_idx = sorted(stt_sel)
        cold_idx = sorted(cold_sel)

        kvt = cfg.resolved_kv_dtype
        self.sram_k = k_all[:, sram_idx, :].clone().to(kvt)
        self.sram_v = v_all[:, sram_idx, :].clone().to(kvt)
        self.sram_pos = list(sram_idx)
        self.sram_cum = importance[sram_idx].clone().to(self.cfg.dtype)
        self.sram_age = torch.zeros(len(sram_idx), device=cfg.device, dtype=cfg.dtype)
        self.sram_grace = torch.zeros(len(sram_idx), dtype=torch.bool)

        self.stt_k = k_all[:, stt_idx, :].clone().to(kvt)
        self.stt_v = v_all[:, stt_idx, :].clone().to(kvt)
        self.stt_pos = list(stt_idx)
        self.stt_last_access = torch.zeros(len(stt_idx), device=cfg.device, dtype=cfg.dtype)
        self.stt_shadow = torch.zeros(len(stt_idx), dtype=torch.bool)
        # Bifurcated STT tokens were never VRAM-resident, so they promote on
        # first candidacy; the re-entry gate only applies to evicted tokens.
        self.stt_was_vram_resident = torch.zeros(len(stt_idx), dtype=torch.bool)
        self.stt_peek_streak = torch.zeros(len(stt_idx), device=cfg.device, dtype=cfg.dtype)

        return {
            "sram": len(sram_idx), "sttram": len(stt_idx),
            "dropped": len(cold_idx), "prompt_len": N,
        }

    # =====================================================================
    # STEP 1 : sketch check (Quest min/max upper bound)
    # =====================================================================
    def _page_bounds(self):
        """Return per-page (min_k, max_k) sketches and token-index slices.

        Fully vectorized: computes min/max for every page in ONE batched
        reduction instead of one GPU call per page (measured to be a
        meaningful share of sketch_check's cost with dozens of pages,
        called every decode step for every layer).
        """
        H, m, D = self.stt_k.shape
        ps = self.cfg.page_size
        if m == 0:
            return []

        num_pages = (m + ps - 1) // ps
        pad = num_pages * ps - m
        dev = self.stt_k.device

        shadow_t = self.stt_shadow.to(dev)
        if pad > 0:
            k_pad = torch.cat(
                [self.stt_k, torch.zeros(H, pad, D, device=dev, dtype=self.stt_k.dtype)], dim=1
            )
            shadow_pad = torch.cat(
                [shadow_t, torch.ones(pad, dtype=torch.bool, device=dev)]  # padding excluded like a shadow row
            )
        else:
            k_pad, shadow_pad = self.stt_k, shadow_t

        k_reshaped = k_pad.view(H, num_pages, ps, D)
        shadow_reshaped = shadow_pad.view(num_pages, ps)
        mask = shadow_reshaped.view(1, num_pages, ps, 1)

        mins = k_reshaped.masked_fill(mask, float("inf")).min(dim=2).values   # (H, num_pages, D)
        maxs = k_reshaped.masked_fill(mask, float("-inf")).max(dim=2).values  # (H, num_pages, D)
        valid_page = (~shadow_reshaped).any(dim=1)  # (num_pages,)

        pages = []
        for pi in torch.where(valid_page)[0].tolist():
            start, end = pi * ps, min((pi + 1) * ps, m)
            pages.append((start, end, mins[:, pi, :], maxs[:, pi, :]))
        return pages

    @staticmethod
    def page_upper_bounds(q, pages):
        """Quest page bound for every page in `pages`, as a (P,) tensor.

        Split out of sketch_check() so the bound is directly testable: the
        formula is the one thing here that must satisfy a hard mathematical
        property (bound >= max_{k in page} q.k), and a test that recomputes
        it locally rather than calling this would pass even if sketch_check
        regressed. See tests/test_sketch_bound.py.

        q     : (H, 1, D) query for this decode step
        pages : list of (start, end, min_k, max_k) from _page_bounds()
        """
        q2 = q.squeeze(1)                                     # (H, D)
        mn_stack = torch.stack([p[2] for p in pages], dim=0)  # (P, H, D)
        mx_stack = torch.stack([p[3] for p in pages], dim=0)  # (P, H, D)

        # Quest's page bound is  sum_d max(q_d * min_d, q_d * max_d)  -- the
        # elementwise maximum is taken PER FEATURE DIMENSION and only then
        # reduced over D. Taking the maximum after reducing over D (i.e.
        # max(sum_d q*min, sum_d q*max)) is a different, much smaller
        # quantity, and is NOT an upper bound at all: for a query with mixed
        # signs the two sums cancel internally, and over 2000 random trials
        # (D=8, page=16) that form fell BELOW the page's true maximum dot
        # product on 60.9% of them (worst case: true max 12.815, correct
        # bound 15.458, the sum-first form 0.787). Since the bound gates
        # which pages are even considered for promotion, underestimating it
        # silently discards the most relevant pages.
        #
        # Batched into one (P, H, D) op rather than one Python-level .item()
        # sync per page (up to ~40+ pages/call at realistic STT-RAM sizes).
        per_dim_max = torch.maximum(q2.unsqueeze(0) * mn_stack,
                                    q2.unsqueeze(0) * mx_stack)  # (P, H, D)
        return per_dim_max.sum(-1).sum(-1)                       # (P,)

    def sketch_check(self, q):
        """
        q : (H, 1, D). Returns list of (start, end) slices that clear the
        cheap sketch bound -- i.e. PEEK/PROMOTION CANDIDATES, not
        automatically promotions. resolve_promotions() below decides which
        of these actually get physically moved to VRAM this step versus
        merely attended to in place (see its docstring). Also returns the
        sketch MAC count.
        """
        H, m, D = self.stt_k.shape
        if m == 0:
            return [], 0

        pages = self._page_bounds()
        if not pages:
            return [], 0
        ub = self.page_upper_bounds(q, pages)   # (P,) -- see that method

        if self.sram_k.shape[1] > 0:
            s_vram = torch.matmul(q, self.sram_k.transpose(-2, -1)).squeeze(1)
            # Threshold against the MEDIAN current VRAM relevance, not the
            # min. The min is almost always the most recently arrived
            # occupant (grace-protected, cum=0, unproven) -- comparing new
            # candidates against that single weakest, still-unproven
            # resident creates a self-reinforcing promote/evict feedback
            # loop (each promotion creates a new "weakest occupant" that
            # the next candidate trivially beats), causing runaway churn
            # that was measured to promote/demote nearly every token in
            # VRAM every single step with ~0% of promotions surviving long
            # enough to be useful. The median is a page's bar against a
            # "typical" resident, not the most fragile one.
            thresh = s_vram.sum(0).median()
        else:
            thresh = torch.tensor(float("-inf"), device=ub.device)

        k = min(self.cfg.promote_top_pages, ub.shape[0])
        top_vals, top_idx = torch.topk(ub, k)
        keep = (top_vals > thresh).tolist()
        top_idx = top_idx.tolist()
        hits = [pages[i][:2] for i, k_ in zip(top_idx, keep) if k_]
        sketch_macs = 2 * H * len(pages) * D
        return hits, sketch_macs

    # =====================================================================
    # STEP 1b : peek vs. promote -- hysteresis on RE-entry only
    # =====================================================================
    _HYSTERESIS_MIN_STREAK = 2  # smallest value that means "more than
    # once" -- the structural minimum for "requires repeated candidacy",
    # not a tuned hyperparameter. See resolve_promotions() docstring.

    def resolve_promotions(self, candidates):
        """Split sketch-check candidates into (promote_now, peek_only).

        Promotion currently pays the full cost of a residency move (VRAM
        slot contention, later write-back) just to let the model glance at
        a token once -- measured to be the dominant source of promote/evict
        churn (near-max promotion rate, most of it single-use). This
        separates VISIBILITY from RESIDENCY:

        - A candidate page that has NEVER been VRAM-resident before
          promotes immediately on its first candidacy -- identical to the
          pre-peek/promote behavior, so a token gets its first chance at
          VRAM exactly as before.
        - A candidate page that WAS evicted from VRAM before (it already
          had its chance and proved, at least once, not durably valuable
          enough to keep) must clear the sketch bar for
          _HYSTERESIS_MIN_STREAK consecutive steps before re-promoting.
          This directly targets the measured churn pattern: promote, one
          glance, evict, immediately re-qualify, repeat.
        - Anything that doesn't clear the promotion bar is still returned
          as a peek candidate: get_peek_kv() + compute_attention() let the
          model attend to it in place this step (real exact attention, not
          just the sketch's upper bound), with no residency change, no
          write, no eviction pressure. Nearly free -- it's HAM already
          being paid for by the sketch check.

        Deliberately NOT a self-tuning threshold (e.g. an ARC-style
        adaptive split): a purely reactive "calibrate the bar from past
        promotions" scheme deadlocks here, since gating pages to a streak
        of 1 forces every recorded promotion to have streak exactly 1,
        which then re-derives a threshold of 1 forever. `2` is left as an
        explicit structural minimum (the smallest value that means
        "repeated" at all) rather than papering over that with a fabricated
        adaptive number; a real adaptive scheme would need a calibration
        phase decoupled from the gating decision, which is future work.
        """
        m = self.stt_k.shape[1]
        if m == 0 or not candidates:
            return [], []

        candidate_idx = set()
        for (s, e) in candidates:
            candidate_idx.update(i for i in range(s, e) if not self.stt_shadow[i])

        in_cand = torch.zeros(m, dtype=torch.bool, device=self.stt_peek_streak.device)
        if candidate_idx:
            in_cand[list(candidate_idx)] = True
        self.stt_peek_streak = torch.where(
            in_cand, self.stt_peek_streak + 1, torch.zeros_like(self.stt_peek_streak)
        )

        promote_hits, peek_hits = [], []
        for (s, e) in candidates:
            real_idx = [i for i in range(s, e) if not self.stt_shadow[i]]
            if not real_idx:
                continue
            needs_gate = any(self.stt_was_vram_resident[i] for i in real_idx)
            streak_here = self.stt_peek_streak[real_idx].min().item()
            if needs_gate and streak_here < self._HYSTERESIS_MIN_STREAK:
                peek_hits.append((s, e))
            else:
                promote_hits.append((s, e))
        return promote_hits, peek_hits

    def get_peek_kv(self, peek_hits):
        """Gather (but do not move) K/V for peek-only candidate pages, for
        use in this step's attention alongside the VRAM working set. No
        residency change, no write, no eviction pressure -- see
        resolve_promotions() docstring.

        Returns (k_peek, v_peek, peek_pos) where peek_pos is each peeked
        token's TRUE ORIGINAL DOCUMENT POSITION (stable identity), not its
        raw array index -- promote() can reindex/shrink the STT arrays
        (non-inclusive mode) between this call and any later use of these
        tokens' indices, so callers must re-resolve position -> current
        index afterward rather than reusing a raw index across that call.
        """
        idx = []
        for (s, e) in peek_hits:
            idx += [i for i in range(s, e) if not self.stt_shadow[i]]
        if not idx:
            return None, None, []
        peek_pos = [self.stt_pos[i] for i in idx]
        return self.stt_k[:, idx, :], self.stt_v[:, idx, :], peek_pos

    # =====================================================================
    # STEP 2 : promote (STT-RAM -> VRAM)
    # =====================================================================
    def promote(self, hits):
        """Move real-victim tokens in hit pages STT-RAM -> VRAM."""
        if not hits:
            self.last_floor_deferred = 0
            return 0

        H, m, D = self.stt_k.shape
        idx = []
        for (s, e) in hits:
            idx += [i for i in range(s, e) if not self.stt_shadow[i]]
        if not idx:
            self.last_floor_deferred = 0
            return 0

        # Live floor (see TieredConfig.stt_live_floor): inclusive promotion
        # converts live rows to shadows, so unchecked mass promotion drains a
        # layer's live pool to zero and trips the cross-layer exposure veto.
        # Cap live-row promotion, hottest-first (callers pass hottest-first),
        # and report the truncated tail as deferred demand rather than
        # dropping it silently. Non-inclusive promotion deletes rows instead
        # of shadowing, so the veto state is unreachable there regardless.
        self.last_floor_deferred = 0
        floor = self.cfg.stt_live_floor
        if self.cfg.inclusive and floor > 0:
            live_before = m - int(self.stt_shadow.sum())
            allowed = max(0, live_before - floor)
            if len(idx) > allowed:
                self.last_floor_deferred = len(idx) - allowed
                idx = idx[:allowed]
                if not idx:
                    return 0

        self.sram_k = torch.cat([self.sram_k, self.stt_k[:, idx, :]], dim=1)
        self.sram_v = torch.cat([self.sram_v, self.stt_v[:, idx, :]], dim=1)
        self.sram_pos += [self.stt_pos[i] for i in idx]
        self.sram_cum = torch.cat(
            [self.sram_cum, torch.zeros(len(idx), device=self.cfg.device, dtype=self.cfg.dtype)]
        )
        self.sram_age = torch.cat(
            [self.sram_age, torch.zeros(len(idx), device=self.cfg.device, dtype=self.cfg.dtype)]
        )
        # Promoted tokens enter with cum=0, which would make them the very
        # next eviction candidate before compute_attention() ever gives them
        # a chance to be attended to. Grace-protect them for exactly one
        # evict_and_demote() pass (cleared at the end of that pass) so they
        # compete on REAL accumulated attention afterward, not a fabricated
        # score. Structurally identical to sink/window protection.
        self.sram_grace = torch.cat(
            [self.sram_grace, torch.ones(len(idx), dtype=torch.bool)]
        )

        # This token is now (or again) VRAM-resident -- mark it so, so a
        # future eviction back to STT-RAM knows to gate its re-promotion
        # (see resolve_promotions). Idempotent for tokens promoted before.
        self.stt_was_vram_resident[idx] = True

        if self.cfg.inclusive:
            self.stt_shadow[idx] = True
            self.stt_last_access[idx] = self._t
        else:
            drop = set(idx)
            keep_idx = [i for i in range(m) if i not in drop]
            self.stt_k = self.stt_k[:, keep_idx, :]
            self.stt_v = self.stt_v[:, keep_idx, :]
            self.stt_pos = [self.stt_pos[i] for i in keep_idx]
            self.stt_last_access = self.stt_last_access[keep_idx]
            self.stt_shadow = self.stt_shadow[keep_idx]
            self.stt_was_vram_resident = self.stt_was_vram_resident[keep_idx]
            self.stt_peek_streak = self.stt_peek_streak[keep_idx]

        return len(idx)

    # =====================================================================
    # STEP 3 : compute attention over VRAM working set
    # =====================================================================
    def compute_attention(self, q, peek_k=None, peek_v=None):
        """Attention over the VRAM working set, optionally extended with
        peeked STT-RAM pages (see resolve_promotions/get_peek_kv) for this
        step only -- peeked tokens contribute to the real output and get
        real exact-attention feedback, without becoming VRAM-resident.
        With peek_k=None this is bit-identical to the pre-peek/promote
        behavior.
        """
        n_sram = self.sram_k.shape[1]
        if peek_k is not None and peek_k.shape[1] > 0:
            full_k = torch.cat([self.sram_k, peek_k], dim=1)
            full_v = torch.cat([self.sram_v, peek_v], dim=1)
        else:
            full_k, full_v = self.sram_k, self.sram_v

        out, attn_w, macs = scaled_dot_product_attention(q, full_k, full_v)
        attn_flat = attn_w.squeeze(1)  # (H, Lk_total)
        self.sram_cum += attn_flat[:, :n_sram].sum(0)
        self.sram_age += 1
        peek_attn = attn_flat[:, n_sram:].sum(0) if full_k.shape[1] > n_sram else None
        return out, macs, peek_attn

    # =====================================================================
    # STEP 4 : evict + demote
    # =====================================================================
    def _hard_protected_mask(self):
        """True where a VRAM token must NEVER be evicted (sink / recent window)."""
        n = len(self.sram_pos)
        if n == 0:
            return torch.zeros(0, dtype=torch.bool)
        cfg = self.cfg
        pos_t = torch.tensor(self.sram_pos)
        max_pos = pos_t.max()
        return (pos_t < cfg.sink_size) | (pos_t > max_pos - cfg.window_size)

    def evict_and_demote(self, new_k, new_v, new_pos):
        cfg = self.cfg
        dropped = 0
        demoted = 0
        writes_saved = 0
        reclaimed = 0
        self.last_floor_protected = 0

        # Add new token(s) to VRAM
        kvt = cfg.resolved_kv_dtype
        self.sram_k = torch.cat([self.sram_k, new_k.to(kvt)], dim=1)
        self.sram_v = torch.cat([self.sram_v, new_v.to(kvt)], dim=1)
        if isinstance(new_pos, (list, tuple)):
            self.sram_pos.extend(new_pos)
            n_new = len(new_pos)
        else:
            self.sram_pos.append(new_pos)
            n_new = 1
        self.sram_cum = torch.cat(
            [self.sram_cum, torch.zeros(n_new, device=cfg.device, dtype=cfg.dtype)]
        )
        self.sram_age = torch.cat(
            [self.sram_age, torch.zeros(n_new, device=cfg.device, dtype=cfg.dtype)]
        )
        # newest token is protected via window, not grace
        self.sram_grace = torch.cat([self.sram_grace, torch.zeros(n_new, dtype=torch.bool)])

        # VRAM overflow -> demote the unprotected token with the lowest
        # AVERAGE attention per attended step (sram_cum / sram_age), not
        # raw cumulative attention. Raw cumulative sum structurally favors
        # whichever token has been resident longest: a token promoted 1
        # step ago cannot out-accumulate one that has been sitting in VRAM
        # for 500 steps even if the newcomer is now clearly more relevant,
        # which was measured to make every promoted token the eviction
        # candidate again the moment its one-step grace period ends --
        # permanent thrashing that starves the write-saving mechanism.
        # Averaging by age compares tokens on relevance-per-step instead of
        # tenure. Ties (age=0, i.e. not yet attended even once) fall back
        # to raw cum via the +1 floor, matching the pre-existing behavior
        # for brand-new/just-promoted tokens.
        #
        # Preference order: ordinary tokens first; if none remain (e.g. a
        # promotion just added more grace-protected tokens than there is
        # room for), fall back to evicting a grace token -- capacity is a
        # hard constraint, grace is only a preference. Sink/window tokens
        # are never evicted.
        # Victim SELECTION is batched: none of the candidates' scores change
        # as others are (conceptually) removed within this call -- sram_cum
        # and sram_age are read-only here, so picking the N lowest-scoring
        # unprotected tokens in one shot gives the IDENTICAL set (and, via a
        # stable sort, identical tie-breaking order) as the old one-at-a-time
        # argmin loop, without rebuilding an O(n) Python mask and reallocating
        # the (H, n, D) VRAM tensor on every single evicted token. Measured
        # to be >90% of this function's wall-clock time before batching
        # (32 layers x up to ~32 evictions/layer/step, each doing a full
        # tensor-copy round trip).
        n_over = len(self.sram_pos) - cfg.sram_capacity
        victims = []
        if n_over > 0:
            hard = self._hard_protected_mask()
            soft = hard | self.sram_grace
            cand = self.sram_cum / self.sram_age.clamp(min=1)

            if cfg.write_aware_lambda > 0 and cfg.inclusive:
                # Backup lookup moved BEFORE selection (it used to be built
                # after): a VRAM token with a live STT shadow demotes for
                # free, so subtract lambda from its eviction score. One
                # O(n) Python-membership pass; the tensor it reads already
                # exists, no rebuild.
                backup_pos = set()
                if self.stt_shadow.numel():
                    for i in torch.where(self.stt_shadow)[0].tolist():
                        backup_pos.add(self.stt_pos[i])
                if backup_pos:
                    has_backup = torch.tensor(
                        [1.0 if p in backup_pos else 0.0 for p in self.sram_pos],
                        device=cand.device, dtype=cand.dtype)
                    cand = cand - cfg.write_aware_lambda * has_backup

            cand_soft = cand.clone()
            cand_soft[soft] = float("inf")
            order = torch.argsort(cand_soft, stable=True)
            n_avail = int(torch.isfinite(cand_soft).sum().item())
            take = min(n_over, n_avail)
            victims = order[:take].tolist()

            remaining = n_over - take
            if remaining > 0:
                # Not enough ordinary candidates (e.g. a promotion just added
                # more grace tokens than there is room for): capacity is a
                # hard constraint, grace is only a preference, so fall back
                # to evicting grace tokens too. Sink/window are never touched.
                cand_hard = cand.clone()
                cand_hard[hard] = float("inf")
                already = torch.zeros(len(self.sram_pos), dtype=torch.bool)
                already[victims] = True
                cand_hard[already] = float("inf")
                order_hard = torch.argsort(cand_hard, stable=True)
                n_hard_avail = int(torch.isfinite(cand_hard).sum().item())
                take_hard = min(remaining, n_hard_avail)
                victims += order_hard[:take_hard].tolist()
                # if still short, everything left is sink/window -- matches
                # the old loop's "break": cannot evict further.

        # Backup lookup: build the pos->index map ONCE (O(m)) instead of
        # calling _find_backup's linear scan once per victim (O(m) each,
        # O(m * victims) total). Safe to compute up front and treat as
        # read-only for this batch: sram_pos entries are unique, so no two
        # victims can share a vpos / contend for the same backup slot.
        backup_by_pos = {}
        if cfg.inclusive and self.stt_shadow.numel():
            shadow_idx0 = torch.where(self.stt_shadow)[0].tolist()
            backup_by_pos = {self.stt_pos[i]: i for i in shadow_idx0}

        no_backup_victims = []
        for victim in victims:
            demoted += 1
            vpos = self.sram_pos[victim]
            back = backup_by_pos.get(vpos, -1)
            if back >= 0:
                self.stt_shadow[back] = False
                self.stt_last_access[back] = self._t
                # Fresh re-candidacy episode -- was_vram_resident is already
                # True (that's why it has a backup), streak must not carry
                # over a stale value frozen from before it became a shadow.
                self.stt_peek_streak[back] = 0
                writes_saved += 1
            else:
                no_backup_victims.append(victim)

        if no_backup_victims:
            # One batched STT append for every victim with no existing
            # backup, instead of one torch.cat (full tensor reallocation)
            # per victim.
            self.stt_k = torch.cat([self.stt_k, self.sram_k[:, no_backup_victims, :]], dim=1)
            self.stt_v = torch.cat([self.stt_v, self.sram_v[:, no_backup_victims, :]], dim=1)
            self.stt_pos += [self.sram_pos[v] for v in no_backup_victims]
            self.stt_last_access = torch.cat([
                self.stt_last_access,
                torch.full((len(no_backup_victims),), float(self._t),
                           device=cfg.device, dtype=cfg.dtype),
            ])
            self.stt_shadow = torch.cat(
                [self.stt_shadow, torch.zeros(len(no_backup_victims), dtype=torch.bool)]
            )
            # These tokens just left VRAM -- any future re-promotion is a
            # RE-entry and must clear the hysteresis gate, not the free pass
            # a genuinely-new candidate gets.
            self.stt_was_vram_resident = torch.cat(
                [self.stt_was_vram_resident, torch.ones(len(no_backup_victims), dtype=torch.bool)]
            )
            self.stt_peek_streak = torch.cat([
                self.stt_peek_streak,
                torch.zeros(len(no_backup_victims), device=cfg.device, dtype=cfg.dtype),
            ])

        if victims:
            keep_mask = torch.ones(len(self.sram_pos), dtype=torch.bool)
            keep_mask[victims] = False
            keep = torch.where(keep_mask)[0].tolist()
            self.sram_k = self.sram_k[:, keep, :]
            self.sram_v = self.sram_v[:, keep, :]
            self.sram_pos = [self.sram_pos[i] for i in keep]
            self.sram_cum = self.sram_cum[keep_mask]
            self.sram_age = self.sram_age[keep_mask]
            self.sram_grace = self.sram_grace[keep_mask]

        # STT-RAM overflow -> reclaim shadows first, then LRU evict.
        #
        # Reclaim priority among shadows is NOT last-access LRU: a shadow's
        # last_access is stamped once at promotion time and never updated
        # while its twin sits happily in VRAM, so LRU-by-age reclaims
        # long-resident (i.e. currently IMPORTANT, high cum-attn) tokens'
        # backups first -- exactly backwards. Instead reclaim the shadow
        # whose live VRAM twin currently has the HIGHEST cumulative
        # attention: that token is the least likely to be demoted again
        # soon, so losing its free backup costs the least expected future
        # writes. A shadow whose twin can't be found (shouldn't happen --
        # invariant) is reclaimed first.
        # Batched for the same reason as the VRAM loop above: a candidate's
        # reclaim/eviction priority here (its live twin's cum-attn for
        # shadows, or last_access for real victims) does not change as
        # other candidates are removed, so the N needed removals can be
        # selected in one shot instead of one Python while-loop iteration
        # (with a full dict rebuild) per removed token.
        n_over = len(self.stt_pos) - cfg.sttram_capacity
        if n_over > 0:
            # One boolean tensor pass instead of two separate Python
            # enumerate() loops over stt_shadow -- with STT-RAM near
            # capacity almost every step, this ran O(sttram_capacity)
            # Python-level work per layer per step and was measured to
            # dominate real wall-clock time once STT-RAM got large.
            shadow_t = self.stt_shadow
            shadow_idx = torch.where(shadow_t)[0].tolist()
            evict_set = []

            if shadow_idx:
                sram_cum_by_pos = dict(zip(self.sram_pos, self.sram_cum.tolist()))
                priority = torch.tensor(
                    [sram_cum_by_pos.get(self.stt_pos[i], float("inf")) for i in shadow_idx]
                )
                order = torch.argsort(priority, descending=True, stable=True)
                take = min(n_over, len(shadow_idx))
                evict_set = [shadow_idx[i] for i in order[:take].tolist()]
                reclaimed += take

            remaining = n_over - len(evict_set)
            if remaining > 0:
                real_idx = torch.where(~shadow_t)[0].tolist()
                la = self.stt_last_access[real_idx]
                order = torch.argsort(la, stable=True)
                # Live floor, evict path: once shadows are reclaimed, raw
                # overflow would eat live rows one per step until some layer
                # goes fully shadowed (and vetoes exposure everywhere). Cap
                # live-row eviction at (live - floor) and drop the freshly
                # appended demoted tail instead — losing one new demotion
                # beats hiding every layer's slow tier for a step, and its
                # write was already billed. If the tail can't cover it, the
                # floor is breached as a last resort (counted; the unit test
                # asserts zero breaches in normal driving).
                floor = self.cfg.stt_live_floor if self.cfg.inclusive else 0
                live = len(real_idx)
                take = min(remaining, max(0, live - floor))
                real_evict = [real_idx[i] for i in order[:take].tolist()]
                dropped += len(real_evict)
                evict_set += real_evict
                short = remaining - len(real_evict)
                if short > 0:
                    taken = set(evict_set)
                    m_now = len(self.stt_pos)
                    n_appended = len(no_backup_victims)
                    tail = [i for i in range(m_now - n_appended, m_now)
                            if 0 <= i < m_now and i not in taken]
                    cover = tail[:short]
                    need = short - len(cover)
                    if need > 0:
                        extra = [r for r in (real_idx[i] for i in order.tolist())
                                 if r not in taken and r not in set(cover)][:need]
                        cover += extra
                    dropped += len(cover)
                    evict_set += cover
                    self.last_floor_protected = len(cover)

            keep_mask = torch.ones(len(self.stt_pos), dtype=torch.bool)
            keep_mask[evict_set] = False
            keep = torch.where(keep_mask)[0].tolist()
            self.stt_k = self.stt_k[:, keep, :]
            self.stt_v = self.stt_v[:, keep, :]
            self.stt_pos = [self.stt_pos[i] for i in keep]
            self.stt_last_access = self.stt_last_access[keep_mask]
            self.stt_shadow = self.stt_shadow[keep_mask]
            self.stt_was_vram_resident = self.stt_was_vram_resident[keep_mask]
            self.stt_peek_streak = self.stt_peek_streak[keep_mask]

        # Grace protection is one-shot: it only shields a just-promoted token
        # through the eviction pass immediately following its own promotion.
        self.sram_grace.fill_(False)

        return demoted, dropped, writes_saved, reclaimed

    # =====================================================================
    # Full decode step orchestrator
    # =====================================================================
    def step(self, q, new_k, new_v, new_pos):
        self._t += 1
        H, D = self.cfg.num_heads, self.cfg.head_dim
        rec = StepRecord(step=self._t)

        # 1. Sketch check -> peek/promotion candidates
        candidates, sketch_macs = self.sketch_check(q)
        rec.sketch_macs = sketch_macs

        # 1b. Hysteresis: first-time candidates promote immediately (as
        # before); re-candidates (previously evicted from VRAM) must clear
        # the bar repeatedly first -- see resolve_promotions() docstring.
        promote_hits, peek_hits = self.resolve_promotions(candidates)

        # 2a. Peek (read-only, no residency change) -- gather still-candidate
        # pages for this step's attention only. Must happen BEFORE promote()
        # below: in non-inclusive mode, promote() removes indices from the
        # STT arrays and everything after it shifts, which would leave
        # peek_hits' (start, end) slices pointing at the wrong -- or
        # out-of-range -- tokens. promote_hits/peek_hits are always disjoint
        # page ranges, so gathering peek data first is always safe.
        peek_k, peek_v, peek_pos = self.get_peek_kv(peek_hits)
        rec.peeked_tokens = len(peek_pos)

        # 2b. Promote (physical move -- VRAM slot + eventual write-back cost)
        rec.promoted_tokens = self.promote(promote_hits)

        # 3. Attention on VRAM working set (+ peeked STT pages, if any)
        out, attn_macs, peek_attn = self.compute_attention(q, peek_k, peek_v)
        rec.attn_macs = attn_macs

        if peek_pos and peek_attn is not None:
            # Real exact-attention feedback (not just the sketch bound):
            # a peeked-but-not-promoted token that actually got attended to
            # is marked freshly relevant, protecting it from LRU-based STT
            # reclaim a bit longer -- reuses the existing stt_last_access
            # mechanism rather than inventing a new signal. Re-resolve
            # position -> current array index since promote() above may
            # have reindexed the STT arrays (non-inclusive mode).
            touched_pos = [p for p, a in zip(peek_pos, peek_attn.tolist()) if a > 0]
            if touched_pos:
                pos_to_idx = {p: i for i, p in enumerate(self.stt_pos)}
                touched_idx = [pos_to_idx[p] for p in touched_pos if p in pos_to_idx]
                if touched_idx:
                    self.stt_last_access[touched_idx] = self._t

        # 4. Evict + demote (also inserts new token)
        demoted, dropped, writes_saved, reclaimed = self.evict_and_demote(new_k, new_v, new_pos)

        rec.demoted_tokens = demoted
        rec.dropped_tokens = dropped
        rec.writes_saved = writes_saved
        rec.backups_reclaimed = reclaimed

        rec.sram_tokens = len(self.sram_pos)
        rec.sttram_tokens = len(self.stt_pos)

        # Cost model
        paid_writes = rec.demoted_tokens - rec.writes_saved
        elems_per_tok = H * D
        lat_p, eng_p = self.cm.promote_cost(rec.promoted_tokens * 2 * elems_per_tok)
        lat_d, eng_d = self.cm.demote_cost(paid_writes * 2 * elems_per_tok)
        # Dropped tokens are freed, not migrated: the only physical work is
        # the STT-RAM read that retires the line.
        lat_dd, eng_dd = self.cm.drop_cost(rec.dropped_tokens * 2 * elems_per_tok)
        lat_vr, eng_vr = self.cm.gpu_vram_read_cost(rec.sram_tokens * 2 * elems_per_tok)
        # Peek: a real STT-RAM read for exact attention, no VRAM write --
        # the whole point is that this is cheaper than promote_cost.
        lat_pk, eng_pk = self.cm.read_latency_us("STT-RAM", rec.peeked_tokens * 2 * elems_per_tok), \
                         self.cm.read_energy_nj("STT-RAM", rec.peeked_tokens * 2 * elems_per_tok)

        n_pages = math.ceil(max(rec.sttram_tokens, 1) / self.cfg.page_size)
        lat_sk, eng_sk = self.cm.sketch_read_cost(2 * n_pages * H * D, tier="STT-RAM")

        rec.lat_sketch_us = lat_sk
        rec.lat_promote_us = lat_p
        rec.lat_peek_us = lat_pk
        rec.lat_attention_us = lat_vr
        rec.lat_demote_us = lat_d + lat_dd
        rec.latency_us = lat_sk + lat_p + lat_pk + lat_vr + lat_d + lat_dd
        rec.energy_nj = eng_sk + eng_p + eng_pk + eng_vr + eng_d + eng_dd

        self.metrics.add(rec)
        return out
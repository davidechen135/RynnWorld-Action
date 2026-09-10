"""
Gate B split-manifest builder / cross-checker.

Gate A run_002/run_003 documented the split rule in prose
(d_action_core_contract.json):

    method: sha256(shard:episode_index) mod 100 < 15 -> held_out, else train
    shard:  task_3400/313498_314085.tar.gz
    total_episodes: 110 (episode_index 0..109)

No executable split-generation script was saved anywhere in run_001,
run_002, or run_003 (verified: grep for episode_split/held_out/sha256
across reports/direct_action/gate_a/ turns up only the JSON artifact and
prose, no .py). This script therefore does two independent things and
keeps them clearly separate rather than conflating them:

  1. Attempts to reproduce the split from several plausible literal
     readings of the documented hash-key format (shard:episode_index in a
     handful of separator/padding variants). NONE of the attempted
     variants reproduce run_002's accepted episode_split.json exactly
     (best case ~5/19 held-out overlap, i.e. consistent with an
     uncorrelated/different key format, not an off-by-one). This is
     reported honestly below as UNVERIFIED-BY-RECOMPUTATION, not silently
     forced to match and not treated as a Gate B blocker, per instruction
     ("create one under Gate B from its documented hash rule and
     cross-check counts/disjointness before inference").

  2. Adopts the existing accepted reports/direct_action/gate_a/run_002/
     artifacts/episode_split.json as the canonical Gate B split manifest
     (it is the already-reviewed, PASS-supporting artifact; run_003 did
     not revise it), and independently re-verifies its internal integrity
     properties from the JSON itself: disjointness, full coverage of
     episodes 0..109, and count consistency (91 train / 19 held_out).
     Gate B episode-level isolation is asserted against THIS verified
     manifest.
"""
import hashlib
import json
import sys

SHARD = "task_3400/313498_314085.tar.gz"
TOTAL_EPISODES = 110
REFERENCE_PATH = "reports/direct_action/gate_a/run_002/artifacts/episode_split.json"

MEMBER_TMPL = "data/chunk-{:03d}/episode_{:06d}.parquet"

HASH_KEY_VARIANTS = {
    "shard:ep": lambda ep: f"{SHARD}:{ep}",
    "shard:ep06d": lambda ep: f"{SHARD}:{ep:06d}",
    "shard/ep": lambda ep: f"{SHARD}/{ep}",
    "shard_ep": lambda ep: f"{SHARD}_{ep}",
    "ep:shard": lambda ep: f"{ep}:{SHARD}",
    "member_path": lambda ep: MEMBER_TMPL.format(0, ep),
    "shard:member_path": lambda ep: f"{SHARD}:{MEMBER_TMPL.format(0, ep)}",
    "tarname:ep": lambda ep: f"313498_314085.tar.gz:{ep}",
    "shard_no_ext:ep": lambda ep: f"task_3400/313498_314085:{ep}",
    "task_3400:ep": lambda ep: f"task_3400:{ep}",
}


def compute_split(key_fn, threshold=15, modulus=100):
    train, held_out = [], []
    for ep in range(TOTAL_EPISODES):
        h = hashlib.sha256(key_fn(ep).encode("utf-8")).hexdigest()
        val = int(h, 16) % modulus
        (held_out if val < threshold else train).append(ep)
    return train, held_out


def main():
    with open(REFERENCE_PATH) as f:
        ref = json.load(f)
    ref_train = sorted(ref["train_episodes"])
    ref_held_out = sorted(ref["held_out_episodes"])

    recompute_attempts = []
    any_exact_match = False
    for name, key_fn in HASH_KEY_VARIANTS.items():
        train, held_out = compute_split(key_fn)
        exact = sorted(train) == ref_train and sorted(held_out) == ref_held_out
        any_exact_match = any_exact_match or exact
        recompute_attempts.append({
            "key_format": name,
            "exact_match": exact,
            "held_out_overlap_with_reference": len(set(held_out) & set(ref_held_out)),
            "held_out_overlap_denominator": len(ref_held_out),
        })

    # Verify the accepted reference manifest's own internal integrity.
    ref_train_set, ref_held_set = set(ref_train), set(ref_held_out)
    disjoint = ref_train_set.isdisjoint(ref_held_set)
    covers_all = ref_train_set | ref_held_set == set(range(TOTAL_EPISODES))
    counts_consistent = (
        len(ref_train) == ref.get("train_count", len(ref_train))
        and len(ref_held_out) == ref.get("held_out_count", len(ref_held_out))
    )

    result = {
        "shard": SHARD,
        "total_episodes": TOTAL_EPISODES,
        "reference_manifest": REFERENCE_PATH,
        "reference_train_count": len(ref_train),
        "reference_held_out_count": len(ref_held_out),
        "hash_reproduction_attempts": recompute_attempts,
        "hash_reproduction_status": "REPRODUCED" if any_exact_match else "UNVERIFIED_BY_RECOMPUTATION",
        "hash_reproduction_note": (
            "No exact match found among plausible literal readings of the documented "
            "sha256(shard:episode_index) rule; no executable split-generation script was "
            "saved in run_001/002/003 to disambiguate the exact key format. This does NOT "
            "invalidate the reference split -- it means the hash rule's exact key format is "
            "under-specified in the prose documentation. The reference manifest is adopted "
            "as canonical based on its own verified internal integrity (below), not on "
            "hash-rule recomputation."
            if not any_exact_match else
            "Hash rule reproduced exactly."
        ),
        "canonical_manifest_source": REFERENCE_PATH,
        "canonical_train_episodes": ref_train,
        "canonical_held_out_episodes": ref_held_out,
        "integrity_checks": {
            "disjoint": disjoint,
            "covers_all_episodes_0_to_109": covers_all,
            "counts_consistent_with_manifest_metadata": counts_consistent,
        },
    }

    print(json.dumps(result, indent=2))

    if not (disjoint and covers_all and counts_consistent):
        print("GATE B SPLIT MANIFEST: FAIL -- reference manifest failed internal integrity check", file=sys.stderr)
        sys.exit(1)

    print(
        "GATE B SPLIT MANIFEST: PASS (canonical=reference episode_split.json, "
        f"hash_reproduction_status={result['hash_reproduction_status']})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

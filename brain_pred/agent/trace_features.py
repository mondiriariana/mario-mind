"""What the agent's written reasoning is about, as brain-encoding bands (CPU).

Reads the replies embed_scans.py --reason saved (<run>.agent_<tag>.replies.jsonl: the agent's own perception,
reasoning and answer at every decision of the participant's plays; nothing about the human) and gives every scan
the reply of its most recent decision (same scan -> decision rule as embed_scans.py). Two bands:
  trace_<tag>     interpretable, 29 columns: reply / perception / reasoning length, number of reasoning steps,
                  the answer's buttons (7) and a parse-failure flag, and log(1 + count) of word families in
                  perception + reasoning: enemies, danger, gaps, obstacles, rewards, jumping, speed, caution,
                  uncertainty, goal, Mario's own state, moving back; plus the same families in reasoning only
                  for enemies / danger / gaps / caution / uncertainty (what it worries about, not only sees)
  trace_<tag>lsa  64-d latent semantic analysis of perception + reasoning (tf-idf over the 3,000 most common
                  words, SVD), vocabulary and SVD fitted on training sessions only (configs/holdout_sessions.json)
Writes $MJG_ROOT/results/brain_encoding/tr_features/<sub>/<run>.trace_<tag>[lsa].npy float32 [n_TR, F] + .mask.npy.

  python trace_features.py --subject sub-03 --tag sft
"""
import argparse, json, os, re, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "..")
sys.path.insert(0, REPO)
from mjg import encoding as E  # noqa: E402

ROOT = os.environ.get("MJG_ROOT", ".")
SCRATCH = os.environ.get("MJG_SCRATCH", os.path.join(ROOT, "scratch"))
TURNS = f"{SCRATCH}/glimmer_scan"
TRF = f"{ROOT}/results/brain_encoding/tr_features"
BUTTONS = ["a", "b", "up", "down", "left", "right", "noop"]
FAMILIES = {
    "enemy": (r"goomba|koopa|enem|turtle|piranha|hammer|bullet|lakitu|spiny|beetle|cheep|bowser|fire ?bar|podoboo"
              r"|blooper"),
    "danger": r"danger|threat|avoid|careful|damage|die|dying|death|lose|kill|harm|risk|collid|collision|hurt",
    "gap": r"gap|pit|hole|fall|edge|cliff|chasm|drop",
    "obstacle": r"pipe|wall|block|brick|stair|obstacle|ledge",
    "reward": r"coin|mushroom|power|star|flower|item|question|bonus|collect",
    "jump": r"jump|leap|hop",
    "speed": r"run|speed|momentum|fast|sprint|dash",
    "caution": r"wait|stop|pause|slow|timing|safe",
    "uncertain": r"maybe|might|perhaps|unclear|uncertain|not sure|possibl|could|seems|likely",
    "goal": r"flag|castle|goal|finish|progress",
    "self": r"small mario|big mario|super mario|fire mario|lives|life",
    "back": r"back|behind|retreat|left",
}
WORRY = ["enemy", "danger", "gap", "caution", "uncertain"]


def compile_families():
    rx = {}
    for k, v in FAMILIES.items():
        rx[k] = re.compile(rf"\b(?:{v})", re.I)
    return rx


RX = compile_families()


def play_id(stem):  # as embed_scans.play_id
    m = re.match(r"(sub-\d+)_(ses-\d+)_task-mario_level-(w\dl\d)_rep-(\d+)", stem)
    return f"{m.group(1)}_{m.group(2)}_{m.group(3)}_rep-{m.group(4)}" if m else stem


def section(reply, name):
    m = re.search(rf"<{name}>(.*?)(?:</{name}>|$)", reply, re.S)
    return m.group(1) if m else ""


def text_of(r):
    """Perception + reasoning of a saved reply, lower case."""
    return (section(r["reply"], "perception") + " " + section(r["reply"], "reasoning")).lower()


def words_of(s):
    return re.findall(r"[a-z]{3,}", s)


def lexical(reply, n_tokens):
    per, rea = section(reply, "perception"), section(reply, "reasoning")
    ans = re.findall(r"<answer>(.*?)</answer>", reply, re.S)
    btn = []
    if ans:
        for b in re.findall(r"'([A-Za-z]+)'", ans[-1]):
            btn.append(b.lower())
    f = [np.log1p(n_tokens), np.log1p(len(per.split())), np.log1p(len(rea.split())),
         len(re.findall(r"^\s*\d+\.", rea, re.M))]
    for b in BUTTONS:
        f.append(float(b in btn))
    f.append(float(not ans))
    for k in FAMILIES:
        f.append(np.log1p(len(RX[k].findall(per + " " + rea))))
    for k in WORRY:
        f.append(np.log1p(len(RX[k].findall(rea))))
    return np.array(f, np.float32)


def load_replies(path):
    """replies.jsonl -> {(play, turn): reply record}."""
    out = {}
    for line in open(path):
        r = json.loads(line)
        out[(r["play"], r["turn"])] = r
    return out


def fit_lsa(train, n_vocab, n_lsa):
    """tf-idf vocabulary + SVD on the training replies -> (tfidf function, mean, components, explained share)."""
    df = {}
    for s in train:
        for w in set(words_of(s)):
            df[w] = df.get(w, 0) + 1
    vocab = {}
    for i, w in enumerate(sorted(df, key=lambda w: -df[w])[:n_vocab]):
        vocab[w] = i
    counts = []
    for w in vocab:  # insertion order = column order
        counts.append(df[w])
    idf = np.log(len(train) / np.array(counts, np.float64))

    def tfidf(docs):
        M = np.zeros((len(docs), len(vocab)), np.float32)
        for i, s in enumerate(docs):
            for w in words_of(s):
                if w in vocab:
                    M[i, vocab[w]] += 1
        M = np.log1p(M) * idf
        return M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-8)
    Mtr = tfidf(train)
    mu = Mtr.mean(0)
    _, sv, Vt = np.linalg.svd(Mtr - mu, full_matrices=False)
    return tfidf, mu, Vt[:n_lsa].T, (sv[:n_lsa] ** 2).sum() / (sv ** 2).sum(), len(vocab)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", default="sub-03")
    ap.add_argument("--tag", default="sft")
    ap.add_argument("--n-lsa", type=int, default=64)
    ap.add_argument("--vocab", type=int, default=3000)
    a = ap.parse_args()
    d = f"{TRF}/{a.subject}"
    n_trs = E.run_trs_from_confounds(a.subject)
    st = E.scan_table(a.subject, n_trs)
    hold = set(json.load(open(os.path.join(REPO, "configs", "holdout_sessions.json")))["holdout"].get(a.subject, []))
    runs = []
    for k in sorted(st.key.unique()):
        if os.path.exists(f"{d}/{k}.agent_{a.tag}.replies.jsonl"):
            runs.append(k)
    print(f"{a.subject}: {len(runs)} runs with replies", flush=True)
    reps = {}
    for k in runs:
        reps[k] = load_replies(f"{d}/{k}.agent_{a.tag}.replies.jsonl")
    train = []  # LSA fitted on training sessions only
    for k in runs:
        if k.split("_")[0] not in hold:
            for r in reps[k].values():
                train.append(text_of(r))
    tfidf, mu, comps, explained, n_vocab = fit_lsa(train, a.vocab, a.n_lsa)
    print(f"  LSA on {len(train)} training replies, vocabulary {n_vocab}, {a.n_lsa} components explain "
          f"{explained:.2f}", flush=True)
    cache = {}
    for k in runs:
        g = st[st.key == k].sort_values("tr")
        T = len(g)
        lex = np.zeros((T, len(BUTTONS) + 5 + len(FAMILIES) + len(WORRY)), np.float32)
        lsa = np.zeros((T, a.n_lsa), np.float32)
        mask = np.zeros(T, bool)
        per_dec = {}
        for u, r in reps[k].items():
            topics = ((tfidf([text_of(r)])[0] - mu) @ comps).astype(np.float32)
            per_dec[u] = (lexical(r["reply"], r["n_tokens"]), topics)
        for r in g.itertuples():
            if not r.stem or r.f1 <= 0:
                continue
            pid = play_id(r.stem)
            if pid not in cache:
                p = f"{TURNS}/{a.subject}/all/{pid}.npz"
                cache.clear()
                cache[pid] = np.load(p)["t"] if os.path.exists(p) else None
            t = cache[pid]
            if t is None or not len(t):
                continue
            i = int(np.searchsorted(t, r.f1, side="left")) - 1  # last turn starting before the scan ends
            if i < 0 or (pid, i) not in per_dec:
                continue
            lex[int(r.tr)], lsa[int(r.tr)] = per_dec[(pid, i)]
            mask[int(r.tr)] = True
        for t2, arr in ((a.tag, lex), (f"{a.tag}lsa", lsa)):
            np.save(f"{d}/{k}.trace_{t2}.npy", arr)
            np.save(f"{d}/{k}.trace_{t2}.mask.npy", mask)
        print(f"  {k}: {mask.sum()}/{T} scans, {len(per_dec)} replies", flush=True)


if __name__ == "__main__":
    main()

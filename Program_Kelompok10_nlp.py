import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, brier_score_loss, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")
#Pembagian acak 80% data latih & 20% data uji 
SEED = 42
TEST_SIZE = 0.2
N_BOOT = 1000 #Jumlah pengulangan bootstrap untuk interval kepercayaan 95%
N_BINS = 10 #probabilitas untuk perhitungan kkalibrasi ECE
NEAR_DUP_THRESHOLD = 0.95 # Batas kemiripan cosine similarity untuk buang brita identik

#  Memuat data
TEXT_CANDIDATES = ["text", "teks", "content", "isi", "berita", "artikel", "narasi", "clean_text", "title", "judul"]
LABEL_CANDIDATES = ["label", "kelas", "class", "kategori", "target", "hoax", "hoaks"]
SOURCE_CANDIDATES = ["source", "sumber", "media", "portal", "site", "situs"]


def read_any(path):
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, on_bad_lines="skip")
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Tidak bisa membaca {path}")


def pick_col(df, candidates, override=None, text_like=False):
    if override:
        return override
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in lower:
            return lower[c]
    if text_like:  # kolom teks terpanjang
        obj = [c for c in df.columns if df[c].dtype == object]
        if obj:
            return max(obj, key=lambda c: df[c].astype(str).str.len().mean())
    return None


def to_bin(v):#Mengonversi jadi biner 1 kalo hoaks dan 0 kalau Valid
    s = str(v).strip().lower()
    if s in ("1", "1.0", "hoax", "hoaks"):
        return 1
    if s in ("0", "0.0", "fakta", "fact", "valid", "non-hoax", "non hoax", "non-hoaks", "bukan hoaks"):
        return 0
    if ("hoa" in s) and not ("non" in s or "bukan" in s):
        return 1
    if any(k in s for k in ("fakta", "fact", "non", "bukan")):
        return 0
    return np.nan


SOURCE_HINTS = {
    "turnbackhoax": r"turnbackhoax|mafindo",
    "kompas": r"kompas\.com|kompas",
    "detik": r"detik\.com|detikcom|detiknews",
    "antara": r"antaranews|\(antara\)|antara news",
}


def infer_source(text):
    t = str(text).lower()[:400]
    for name, pat in SOURCE_HINTS.items():
        if re.search(pat, t):
            return name
    return "tidak_diketahui"


def load_data(path, text_col=None, label_col=None, source_col=None):
    path = Path(path)
    files = sorted([p for p in path.rglob("*") if p.suffix.lower() in (".csv", ".xlsx", ".xls")]) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"Tidak ada file data di {path}")
    parts = []
    for f in files:
        d = read_any(f)
        tc = pick_col(d, TEXT_CANDIDATES, text_col, text_like=True)
        lc = pick_col(d, LABEL_CANDIDATES, label_col)
        sc = pick_col(d, SOURCE_CANDIDATES, source_col)
        print(f"[muat] {f.name}: {len(d)} baris | kolom={list(d.columns)} | teks='{tc}', label='{lc}', sumber='{sc}'")
        if tc is None or lc is None:
            raise ValueError(f"{f.name}: kolom teks/label tidak ditemukan. Gunakan --text-col/--label-col.")
        out = pd.DataFrame({"text_raw": d[tc].astype(str), "y": d[lc].map(to_bin)})
        out["source"] = d[sc].astype(str).str.lower() if sc else f.stem.lower()
        parts.append(out)
    df = pd.concat(parts, ignore_index=True)
    bad = df["y"].isna().sum()
    if bad:
        print(f"[peringatan] {bad} baris dengan label tidak dikenali dibuang. Periksa fungsi to_bin().")
    df = df.dropna(subset=["y"])
    df["y"] = df["y"].astype(int)
    df = df[df["text_raw"].str.strip().str.len() > 0].reset_index(drop=True)
    if df["source"].nunique() == 1:
        print("[info] Hanya satu sumber terdeteksi; sumber diperkirakan dari isi teks (heuristik).")
        df["source"] = df["text_raw"].map(infer_source)
    return df


#  Audit & pembersihan artefak sumber ( Mendeteksi kebocoran yang sering buat akurasi model tampak semu)

CITY = (r"jakarta|bandung|surabaya|semarang|yogyakarta|medan|makassar|denpasar|palembang|bogor|depok|bekasi|"
        r"tangerang|malang|solo|surakarta|batam|padang|pekanbaru|manado|balikpapan|samarinda|banjarmasin|"
        r"pontianak|mataram|kupang|jayapura|ambon|palu|kendari|banda aceh")
DETIK = r"detik(?:\.com|com|news|finance|inet|health|sport|sports|food|edu|travel|oto|hot|jabar|jatim|jateng|sumut|bali)"
# (nama aturan, pola). Sengaja TIDAK menghapus kata "antara"/"detik" polos karena bermakna biasa dalam bahasa Indonesia.
RULES = [
    ("url", r"https?://\S+|www\.\S+"),
    ("dateline_media", rf"^\s*(?:{CITY})?[\s,\.]*\(?(?:antara|kompas\.com|kompas\.id|{DETIK})\)?\s*[-–—:]+\s*"),
    ("dateline_kota", rf"^\s*(?:{CITY})\s*(?:,|\()?[^-–—\n]{{0,30}}[-–—]\s+"),
    ("label_verifikasi", r"\[\s*(?:hoax|hoaks|salah|disinformasi|misinformasi|penipuan|fitnah|klarifikasi|fakta|valid|benar)\s*\]"),
    ("nama_media", rf"\b(?:antaranews(?:\.com)?|antara news|kompas(?:\.com|\.id)?|{DETIK}|turnbackhoax(?:\.id)?|mafindo)\b"),
    ("templat_turnbackhoax", r"\b(?:kategori|narasi|penjelasan|referensi|hasil periksa fakta|hasil cek fakta|kesimpulan)\s*:"),
]
COMPILED = [(n, re.compile(p, re.IGNORECASE)) for n, p in RULES]


def clean_text(t):
    for _, rx in COMPILED:
        t = rx.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def audit_and_clean(df):
    print("\n=== AUDIT SUMBER ===")
    ct = pd.crosstab(df["source"], df["y"].map({0: "Fakta", 1: "Hoaks"}), margins=True)
    print(ct)
    rows = []
    for name, rx in COMPILED:
        hit = df["text_raw"].map(lambda t: bool(rx.search(t)))
        for src, n in df[hit].groupby("source").size().items():
            rows.append({"aturan": name, "sumber": src, "teks_terdampak": int(n), "dari_total": int((df["source"] == src).sum())})
    audit = pd.DataFrame(rows)
    df = df.copy()
    df["text_clean"] = df["text_raw"].map(clean_text)
    return df, ct, audit

#Membuang teks bersih dibawah 5 karakter dan duplikat persis, Hitung matriks kempiripan cos TF-IDF
def dedup(df, do_near=True):
    n0 = len(df)
    df = df[df["text_clean"].str.len() >= 5].drop_duplicates(subset="text_clean").reset_index(drop=True)
    n_exact = n0 - len(df)
    n_near = 0
    if do_near and len(df) > 1:
        X = TfidfVectorizer(min_df=2, max_features=50000).fit_transform(df["text_clean"])
        drop = np.zeros(len(df), bool)
        step = 500
        for s in range(0, len(df), step):
            S = (X[s:s + step] @ X.T).tocsr()
            for i in range(S.shape[0]):
                gi = s + i
                if drop[gi]:
                    continue
                cols = S.indices[S.indptr[i]:S.indptr[i + 1]]
                vals = S.data[S.indptr[i]:S.indptr[i + 1]]
                drop[cols[(vals >= NEAR_DUP_THRESHOLD) & (cols > gi)]] = True
        n_near = int(drop.sum())
        df = df[~drop].reset_index(drop=True)
    print(f"[dedup] identik dibuang: {n_exact} | hampir identik (cos >= {NEAR_DUP_THRESHOLD}) dibuang: {n_near} | sisa: {len(df)}")
    return df

#Model & metrik

def make_pipe(max_features=1000, ngram=(1, 1), C=1.0):
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=ngram)),
        ("svm", LinearSVC(C=C, random_state=SEED, max_iter=5000)),
    ])


def metrics(y, p): #Hitung TP, TN, FP,FN, AKURASI , PRESISI, RECALL, F-1 SCORE DAN RECALL FAKTA
    y, p = np.asarray(y), np.asarray(p)
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="binary", pos_label=1, zero_division=0)
    tn = int(((y == 0) & (p == 0)).sum()); fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum()); tp = int(((y == 1) & (p == 1)).sum())
    return {"n": len(y), "akurasi": 100 * accuracy_score(y, p), "presisi": 100 * pr, "recall": 100 * rc, "f1": 100 * f1,
            "recall_fakta": 100 * tn / max(tn + fp, 1), "TP": tp, "TN": tn, "FP": fp, "FN": fn}
#Mengitung rentang kepercayaan 95%
def boot_ci(y, p, n_boot=N_BOOT):
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(SEED)
    correct = (y == p).astype(float)
    accs = [correct[rng.integers(0, len(y), len(y))].mean() for _ in range(n_boot)]
    return 100 * np.percentile(accs, 2.5), 100 * np.percentile(accs, 97.5)

def cv_acc(pipe, X, y):
    s = cross_val_score(pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring="accuracy", n_jobs=-1)
    return 100 * s.mean(), 100 * s.std()

#Hitung ECE serta titik kootdinat grafik berdasakan confidence kelas prediksi
def ece_score(y, p_hoax, n_bins=N_BINS):
    y = np.asarray(y); p_hoax = np.asarray(p_hoax)
    pred = (p_hoax >= 0.5).astype(int)
    conf = np.where(pred == 1, p_hoax, 1 - p_hoax)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0.5 else (conf >= lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def reliability_points(y, p_hoax, n_bins=N_BINS):
    y = np.asarray(y); p_hoax = np.asarray(p_hoax)
    pred = (p_hoax >= 0.5).astype(int)
    conf = np.where(pred == 1, p_hoax, 1 - p_hoax)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0.5 else (conf >= lo) & (conf <= hi)
        if m.sum() >= 5:
            xs.append(conf[m].mean()); ys.append(correct[m].mean())
    return xs, ys

# Membandingkan probabilitas mentah sigmoid naif, platt scaling dan isotonic regression lalu menghitung ECE dan Brier Score
def calibration_models(pipe_fitted, Xtr_text, ytr):
    """Kalibrasi Platt & isotonik. TF-IDF dipasang pada data latih; kalibrator memakai validasi silang 5-lipat pada data latih."""
    vec = pipe_fitted.named_steps["tfidf"]
    svm = pipe_fitted.named_steps["svm"]
    Xtr = vec.transform(Xtr_text)
    out = {}
    for name, method in (("Platt", "sigmoid"), ("Isotonik", "isotonic")):
        kw = dict(method=method, cv=5)
        try:
            cal = CalibratedClassifierCV(estimator=LinearSVC(C=svm.C, random_state=SEED, max_iter=5000), **kw)
        except TypeError:  # scikit-learn < 1.2
            cal = CalibratedClassifierCV(base_estimator=LinearSVC(C=svm.C, random_state=SEED, max_iter=5000), **kw)
        out[name] = cal.fit(Xtr, ytr)
    return vec, out


def p_raw(pipe, texts):
    f = pipe.decision_function(texts)
    return 1 / (1 + np.exp(-f))  # sigmoid naif dari skor SVM (tanpa kalibrasi)


def calib_report(y, texts, pipe, vec, cals, tag):
    ps = {"SVM mentah (sigmoid naif)": p_raw(pipe, texts)}
    Xt = vec.transform(texts)
    for name, c in cals.items():
        ps[name] = c.predict_proba(Xt)[:, 1]
    rows = [{"data": tag, "metode": k, "ECE": ece_score(y, p), "Brier": brier_score_loss(y, p)} for k, p in ps.items()]
    return rows, ps


# Fitur Non Isi ( Menguji seberapa akurats hoaks/ fakta bisa ditebak tampa baca isi topik berita secara menyeluruh)
def nonContent_features(texts):
    rows = []
    for t in texts:
        w = t.split(); n = max(len(t), 1)
        rows.append([len(t), len(w), sum(c in ".,;:!?\"'()-" for c in t) / n, sum(c.isupper() for c in t) / n,
                     sum(c.isdigit() for c in t) / n, np.mean([len(x) for x in w]) if w else 0,
                     t.count(".") + t.count("!") + t.count("?"), t.count("!"), t.count("?")])
    return np.array(rows, dtype=float)


def head_tail(t, k=5):
    w = t.split()
    return " ".join(["H_" + x for x in w[:k]] + ["T_" + x for x in w[-k:]])


def scenario_E(df, col, idx_tr, idx_te, tag):
    y = df["y"].values
    res = {}
    Xf = nonContent_features(df[col])
    m = HistGradientBoostingClassifier(random_state=SEED).fit(Xf[idx_tr], y[idx_tr])
    res[f"D. Probe fitur non-isi  ({tag})"] = metrics(y[idx_te], m.predict(Xf[idx_te]))
    ht = df[col].map(head_tail)
    p = make_pipe(max_features=20000).fit(ht.iloc[idx_tr], y[idx_tr])
    res[f"E. Probe 5 kata awal/akhir  ({tag})"] = metrics(y[idx_te], p.predict(ht.iloc[idx_te]))
    return res


#Main 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"C:\Users\MSI\Downloads\uas_nlp\dataset")
    ap.add_argument("--external", default=None)
    ap.add_argument("--out", default="hasil")
    ap.add_argument("--text-col"); ap.add_argument("--label-col"); ap.add_argument("--source-col")
    ap.add_argument("--quick", action="store_true", help="lewati tuning hiperparameter")
    ap.add_argument("--no-near-dup", action="store_true", help="lewati deteksi hampir-identik")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)


    df, crosstab, audit = audit_and_clean(load_data(a.data, a.text_col, a.label_col, a.source_col))
    crosstab.to_csv(out / "audit_label_x_sumber.csv"); audit.to_csv(out / "audit_artefak.csv", index=False)
    print("\n=== ARTEFAK TERDETEKSI (aturan x sumber) ===")
    print(audit.to_string(index=False) if len(audit) else "(tidak ada yang cocok; periksa RULES)")

    df_raw = df.reset_index(drop=True)
    df_cln = dedup(df, do_near=not a.no_near_dup)
    table, extra = [], {}

    # A = BASELINE ACAK (80:20)
    y = df_raw["y"].values
    itr, ite = train_test_split(np.arange(len(df_raw)), test_size=TEST_SIZE, stratify=y, random_state=SEED)
    A = make_pipe().fit(df_raw["text_raw"].iloc[itr], y[itr])
    pA = A.predict(df_raw["text_raw"].iloc[ite]); mA = metrics(y[ite], pA)
    mA["ci"] = boot_ci(y[ite], pA); mA["cv"] = cv_acc(make_pipe(), df_raw["text_raw"], y)
    table.append(("A. Baseline acak", mA))

    #B = SPLIT 80:20 PADA TEXT_CLEAN. MODEL B PAKAI PARAMETER STANDAR SEDANGKAN B* CARI KOMBINASI LEWAT GRIDSEARCH CV
    yc = df_cln["y"].values
    jtr, jte = train_test_split(np.arange(len(df_cln)), test_size=TEST_SIZE, stratify=yc, random_state=SEED)
    tr_txt, te_txt = df_cln["text_clean"].iloc[jtr], df_cln["text_clean"].iloc[jte]
    B = make_pipe().fit(tr_txt, yc[jtr])
    pB = B.predict(te_txt); mB = metrics(yc[jte], pB)
    mB["ci"] = boot_ci(yc[jte], pB); mB["cv"] = cv_acc(make_pipe(), df_cln["text_clean"], yc)
    table.append(("B. Artefak dibersihkan (1000 fitur, C=1)", mB))

    best = dict(max_features=1000, ngram=(1, 1), C=1.0)
    if not a.quick:
        grid = {"tfidf__max_features": [1000, 20000], "tfidf__ngram_range": [(1, 1), (1, 2)], "svm__C": [0.1, 1, 10]}
        gs = GridSearchCV(make_pipe(), grid, cv=StratifiedKFold(3, shuffle=True, random_state=SEED), scoring="accuracy", n_jobs=-1)
        gs.fit(tr_txt, yc[jtr])
        best = dict(max_features=gs.best_params_["tfidf__max_features"], ngram=gs.best_params_["tfidf__ngram_range"], C=gs.best_params_["svm__C"])
        print(f"[tuning] parameter terbaik (validasi silang pada data latih): {best}")
    Bs = make_pipe(**best).fit(tr_txt, yc[jtr])
    pBs = Bs.predict(te_txt); mBs = metrics(yc[jte], pBs); mBs["ci"] = boot_ci(yc[jte], pBs)
    table.append((f"B*. Artefak dibersihkan + tuning {best}", mBs))
    extra["param_terbaik"] = {k: str(v) for k, v in best.items()}

    # kata paling berpengaruh (untuk mencari artefak yang masih tersisa)
    names = np.array(Bs.named_steps["tfidf"].get_feature_names_out()); coef = Bs.named_steps["svm"].coef_.ravel()
    o = np.argsort(coef)
    top = pd.DataFrame({"kelas": ["Hoaks"] * 40 + ["Fakta"] * 40,
                        "kata": list(names[o[::-1][:40]]) + list(names[o[:40]]),
                        "koefisien": list(coef[o[::-1][:40]]) + list(coef[o[:40]])})
    top.to_csv(out / "top_fitur.csv", index=False)
    print("\n=== 15 KATA TERKUAT PER KELAS (periksa manual: nama media/format = artefak tersisa) ===")
    print("Hoaks:", ", ".join(top[top.kelas == "Hoaks"].kata[:15])); print("Fakta:", ", ".join(top[top.kelas == "Fakta"].kata[:15]))

    # C = UJI LINTAS SUMBER ( TAHAN SATU SUMBER BERITA UNTUK PENGUJIAN, DAN LATIHAN DATA LAINNYA DAN SIMPEN KE SKENARIO_C_DETAIL.CSV)
    fact_src = sorted(df_cln.loc[df_cln.y == 0, "source"].unique())
    rows_c = []
    if len(fact_src) >= 2:
        hx = np.where(df_cln.y.values == 1)[0]
        hx_tr, hx_te = train_test_split(hx, test_size=TEST_SIZE, random_state=SEED)
        src_te = df_cln["source"].iloc[jte].values
        for s in fact_src:
            f_te = np.where((df_cln.y.values == 0) & (df_cln.source.values == s))[0]
            f_tr = np.where((df_cln.y.values == 0) & (df_cln.source.values != s))[0]
            tr, te = np.concatenate([f_tr, hx_tr]), np.concatenate([f_te, hx_te])
            p = make_pipe(**best).fit(df_cln["text_clean"].iloc[tr], yc[tr])
            m = metrics(yc[te], p.predict(df_cln["text_clean"].iloc[te]))
            in_dist = (src_te == s) & (yc[jte] == 0)  # recall fakta sumber s bila sumber itu ikut dilatih (model B*)
            m["recall_fakta_acak"] = 100 * (pBs[in_dist] == 0).mean() if in_dist.any() else np.nan
            m["sumber_ditahan"] = s; rows_c.append(m)
        pd.DataFrame(rows_c).to_csv(out / "skenario_C_detail.csv", index=False)
        mC = {k: float(np.mean([r[k] for r in rows_c])) for k in ("akurasi", "presisi", "recall", "f1", "recall_fakta")}
        mC["n"] = int(np.mean([r["n"] for r in rows_c])); table.append(("C. Uji lintas-sumber (rerata tiap sumber fakta ditahan)", mC))
        print("\n=== SKENARIO C: recall Fakta pada sumber yang ditahan vs pembagian acak ===")
        for r in rows_c:
            print(f"  {r['sumber_ditahan']:<14} ditahan: {r['recall_fakta']:.1f}%  | acak: {r['recall_fakta_acak']:.1f}%")
    else:
        print("[lewati C] butuh minimal 2 sumber berlabel Fakta")

    #KALIBRASI B* PAKAI PLATT & ISOTONIK
    vec, cals = calibration_models(Bs, tr_txt, yc[jtr])
    cal_rows, ps_B = calib_report(yc[jte], te_txt, Bs, vec, cals, "B* (uji internal)")

    #UJI EKSTERNAL 
    ext, ps_D = None, None
    if a.external and Path(a.external).exists():
        ext = read_any(a.external)
        tc = pick_col(ext, TEXT_CANDIDATES, None, text_like=True); lc = pick_col(ext, LABEL_CANDIDATES)
        ext = pd.DataFrame({"t": ext[tc].astype(str).map(clean_text), "y": ext[lc].map(to_bin)}).dropna()
        ext["y"] = ext["y"].astype(int)
        pD = Bs.predict(ext["t"]); mD = metrics(ext["y"], pD); mD["ci"] = boot_ci(ext["y"], pD)
        table.append(("X. Uji eksternal (media sosial, opsional)", mD))
        rD, ps_D = calib_report(ext["y"].values, ext["t"], Bs, vec, cals, "X (eksternal)"); cal_rows += rD
        print(f"\n[X] {len(ext)} teks eksternal; proporsi hoaks = {ext['y'].mean():.2f}")
    else:
        print("\n[lewati X] file --external tidak diberikan")

    # MENJALANKAN UJI FITUR NON ISI PADA DATA MENTAH DAN BERSIH 
    E = {}
    E.update(scenario_E(df_raw, "text_raw", itr, ite, "data mentah"))
    E.update(scenario_E(df_cln, "text_clean", jtr, jte, "data bersih"))
    for k, m in E.items():
        table.append((k, m))
    maj = 100 * max(yc[jte].mean(), 1 - yc[jte].mean())
    extra["kelas_mayoritas_uji_%"] = maj

    # ---- Susun Tabel 4 ----
    accA = mA["akurasi"]
    cal_pl = {(r["data"], r["metode"]): r for r in cal_rows}
    t4 = []
    for name, m in table:
        row = {"Skenario": name, "n_uji": m["n"], "Akurasi (%)": round(m["akurasi"], 2), "Presisi (%)": round(m["presisi"], 2),
               "Recall (%)": round(m["recall"], 2), "F1 (%)": round(m["f1"], 2),
               "Recall Fakta (%)": round(m["recall_fakta"], 2),
               "Penurunan Akurasi vs A (%)": round(100 * (accA - m["akurasi"]) / accA, 2)}
        if "ci" in m: row["CI95 Akurasi"] = f"{m['ci'][0]:.1f}-{m['ci'][1]:.1f}"
        if "cv" in m: row["CV5 Akurasi"] = f"{m['cv'][0]:.2f} ± {m['cv'][1]:.2f}"
        if name.startswith("B*"):
            r = cal_pl[("B* (uji internal)", "Platt")]; row["ECE (Platt)"] = round(r["ECE"], 4); row["Brier (Platt)"] = round(r["Brier"], 4)
        if name.startswith("X") and ("X (eksternal)", "Platt") in cal_pl:
            r = cal_pl[("X (eksternal)", "Platt")]; row["ECE (Platt)"] = round(r["ECE"], 4); row["Brier (Platt)"] = round(r["Brier"], 4)
        t4.append(row)
    T4 = pd.DataFrame(t4); T4.to_csv(out / "tabel4_hasil.csv", index=False)
    C = pd.DataFrame(cal_rows).round(4); C.to_csv(out / "tabel_kalibrasi.csv", index=False)
    print("\n=== TABEL 4 ===\n" + T4.to_string(index=False)); print("\n=== KALIBRASI ===\n" + C.to_string(index=False))
    print(f"\nProporsi kelas mayoritas pada data uji bersih: {maj:.1f}%  (acuan untuk Skenario E)")

    #DIAGRAM
    panels = [("B* (uji internal)", yc[jte], ps_B)]
    if ps_D is not None:
        panels.append(("D (eksternal)", ext["y"].values, ps_D))
    fig, axs = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.6), squeeze=False)
    for ax, (title, yy, ps) in zip(axs[0], panels):
        ax.plot([0.5, 1], [0.5, 1], "k--", lw=1, label="Kalibrasi sempurna")
        for k, p in ps.items():
            xs, ys = reliability_points(yy, p)
            ax.plot(xs, ys, marker="o", ms=4, label=k)
        ax.set_xlabel("Rata-rata confidence"); ax.set_ylabel("Akurasi"); ax.set_title(title); ax.legend(fontsize=7); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(out / "reliability_diagram.png", dpi=200); plt.close()

    extra["jumlah_data_mentah"] = len(df_raw); extra["jumlah_data_setelah_dedup"] = len(df_cln)
    (out / "ringkasan.json").write_text(json.dumps(extra, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"\nSelesai. Hasil ada di di: {out.resolve()}")


if __name__ == "__main__":
    main()

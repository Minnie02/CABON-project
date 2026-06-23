# step2_full_SD_spacer_L9_binding_then_orthogonality.py
# 목적:
# - Step 1에서 선정한 spacer length = 9bp 기준으로
# - SD 6bp ATGC 전체 조합(4^6) × spacer 9bp ATGC 전체 조합(4^9)을 전수조사
# - 1차: binding energy [-9, -7.5] 후보만 저장
# - 2차: binding 후보에 대해 직교성 5가지 필터 적용
#
# 실행:
#   python step2_full_SD_spacer_L9_binding_then_orthogonality.py
#
# 주의:
# - 전체 경우의 수는 4,096 × 262,144 = 1,073,741,824개입니다.
# - 시간이 오래 걸릴 수 있습니다.
# - 중간 결과(binding 후보)는 CSV로 스트리밍 저장하므로 메모리를 크게 쓰지 않습니다.

import os
import csv
import math
from itertools import product
from multiprocessing import Pool, cpu_count
from collections import Counter
from typing import Dict, Iterable, List, Tuple

import RNA


# ============================================================
# 0) 사용자 설정
# ============================================================
SD_LEN = 6
SPACER_LEN = 9

ALPHABET = ("A", "T", "G", "C")

# Binding target
DG_BIND_MIN = -9.0
DG_BIND_MAX = -7.5
DG_BIND_CENTER = (DG_BIND_MIN + DG_BIND_MAX) / 2  # -8.25

# Orthogonality target
DG_ORTHO_MIN = 0.0  # 반드시 > 0

# 교수님 피드백 반영:
# SD 이후 15bp window 사용
AFTER_SD_FIXED_LEN = 15

# SD 조건: ATGC 전체 조합
# 이전 GC 20~40% 제한 없음
USE_SD_GC_FILTER = False

# 입력 GFP 파일 경로
GFP_DNA_PATH = r"C:\Users\rhtmd\OneDrive\바탕 화면\SD design 25-12\4월 9일 미팅이후\gfp와 UTR등의 시퀀스\o-SD test gfp.dna"

# 출력 폴더
OUT_DIR = "step2_full_SD_spacer_L9_binding_then_orthogonality_out"
os.makedirs(OUT_DIR, exist_ok=True)

BINDING_CANDIDATES_CSV = os.path.join(OUT_DIR, "binding_range_candidates_L9_full.csv")
ORTHO_EVALUATED_CSV = os.path.join(OUT_DIR, "orthogonality_evaluated_L9_full.csv")
PASS_ALL_CSV = os.path.join(OUT_DIR, "pass_all_candidates_L9_full.csv")
SUMMARY_CSV = os.path.join(OUT_DIR, "summary_L9_full.csv")
FIRST_FAIL_CSV = os.path.join(OUT_DIR, "first_fail_summary_L9_full.csv")


# ============================================================
# 1) 고정 서열
# ============================================================
UTR_DNA = "ATATAGGCATAGCGCACAGACAGATAAAAATTACAGAGTACACAACATCC"  # 50bp
AU_RICH_DNA = "TTAATTAA"  # 8bp
UP15_DNA = (UTR_DNA + AU_RICH_DNA)[-15:]

WT_SD_DNA = "AGGAGG"

# 기존 코드 기준:
# WT spacer 9bp = GAAACAGCT
# WT-SDsp12는 WT_SD 6bp + WT spacer 앞 6bp = 12bp로 정의
WT_SPACER_9_DNA = "GAAACAGCT"
WT_SPACER_FOR_SDSP12_DNA = WT_SPACER_9_DNA[:6]  # GAAACA
WT_SDSP12_DNA = WT_SD_DNA + WT_SPACER_FOR_SDSP12_DNA

WT_ASD_6_DNA = "CTCCTT"
WT_ASD_12_DNA = "ATCACCTCCTTA"


# ============================================================
# 2) 유틸 함수
# ============================================================
def clean_dna(seq: str) -> str:
    return "".join(ch for ch in seq.upper() if ch in "ATGC")


def dna_to_rna(seq: str) -> str:
    return clean_dna(seq).replace("T", "U")


def revcomp_dna(seq: str) -> str:
    comp = {"A": "T", "T": "A", "G": "C", "C": "G"}
    return "".join(comp[b] for b in reversed(clean_dna(seq)))


def duplex_dg_dna(a_dna: str, b_dna: str) -> float:
    a_rna = dna_to_rna(a_dna)
    b_rna = dna_to_rna(b_dna)
    d = RNA.duplexfold(a_rna, b_rna)
    return float(d.energy)


def read_gfp_cds(path: str) -> str:
    """교수님이 준 GFP .dna 파일에서 sequence만 읽어온다."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"GFP file not found: {path}")

    text = open(path, "r", encoding="utf-8", errors="ignore").read()
    seq = clean_dna(text)

    if len(seq) < 100:
        raise ValueError("GFP sequence seems too short. Check o-SD test gfp.dna file.")

    # 가능하면 GFP 시작 motif를 찾아서 그 이후를 사용
    # 찾지 못하면 전체 sequence를 그대로 사용
    motif = "ATGAGCAAAGGTGAAGAACTGTTTACCG"
    idx = seq.find(motif)
    if idx != -1:
        return seq[idx:]

    # 일반적인 ATG부터 시작하도록 보정
    idx = seq.find("ATG")
    if idx != -1:
        return seq[idx:]

    return seq


GFP_CDS_DNA = read_gfp_cds(GFP_DNA_PATH)


def make_all_sd_list() -> List[str]:
    return ["".join(t) for t in product(ALPHABET, repeat=SD_LEN)]


def make_spacer_iter() -> Iterable[str]:
    for t in product(ALPHABET, repeat=SPACER_LEN):
        yield "".join(t)


def build_after_sd_15(spacer_dna: str) -> str:
    """
    SD 이후 15bp = spacer + GFP 앞부분.
    spacer 9bp일 때: spacer 9bp + GFP 앞 6bp = 15bp
    """
    after = (spacer_dna + GFP_CDS_DNA)[:AFTER_SD_FIXED_LEN]
    if len(after) != AFTER_SD_FIXED_LEN:
        raise ValueError("after_SD_15 length error")
    return after


def build_context21(sd_dna: str, spacer_dna: str) -> str:
    """
    SD 6bp + SD 이후 15bp = 총 21bp
    """
    return sd_dna + build_after_sd_15(spacer_dna)


def build_osdsp12(sd_dna: str, spacer_dna: str) -> str:
    """
    O-SDsp12 = SD 시작점 기준 12bp.
    spacer가 9bp이므로 SD 6bp + spacer 앞 6bp.
    """
    return (sd_dna + spacer_dna)[:12]


def row_fieldnames_base() -> List[str]:
    return [
        "sd_dna",
        "spacer_dna",
        "spacer_len",
        "osdsp12_dna",
        "oasd12_dna",
        "oasd6_dna",
        "context21_dna",
        "after_sd_15_dna",
        "dg_bind_oasd12__context21",
        "dg_bind_oasd12__osdsp12_reference",
    ]


def ortho_fieldnames_extra() -> List[str]:
    return [
        "dg_ortho_wtasd12__up15_sd_spacer",
        "dg_ortho_oasd12__wt_sdsp12",
        "dg_ortho_osdsp12__wt_asd12",
        "dg_ortho_osd6__wt_asd6",
        "dg_ortho_oasd6__wt_sd6",
        "pass_ortho_ctx",
        "pass_ortho_oasd12_wt_sdsp12",
        "pass_ortho_osdsp12_wt_asd12",
        "pass_ortho_osd6_wt_asd6",
        "pass_ortho_oasd6_wt_sd6",
        "pass_all",
        "first_fail",
    ]


# ============================================================
# 3) 1차 screening: binding만 계산
# ============================================================
def binding_worker(task: Tuple[str, str]) -> Dict:
    sd_dna, spacer_dna = task

    osdsp12_dna = build_osdsp12(sd_dna, spacer_dna)
    oasd12_dna = revcomp_dna(osdsp12_dna)
    oasd6_dna = revcomp_dna(sd_dna)

    context21_dna = build_context21(sd_dna, spacer_dna)
    after_sd_15_dna = build_after_sd_15(spacer_dna)

    # 교수님 피드백 반영 main binding:
    # O-ASD12 ↔ SD 6bp + after-SD 15bp context
    dg_bind_context21 = duplex_dg_dna(oasd12_dna, context21_dna)

    # 기존 기준 참고값:
    # O-ASD12 ↔ O-SDsp12
    dg_bind_ref = duplex_dg_dna(oasd12_dna, osdsp12_dna)

    return {
        "sd_dna": sd_dna,
        "spacer_dna": spacer_dna,
        "spacer_len": SPACER_LEN,
        "osdsp12_dna": osdsp12_dna,
        "oasd12_dna": oasd12_dna,
        "oasd6_dna": oasd6_dna,
        "context21_dna": context21_dna,
        "after_sd_15_dna": after_sd_15_dna,
        "dg_bind_oasd12__context21": dg_bind_context21,
        "dg_bind_oasd12__osdsp12_reference": dg_bind_ref,
    }


def task_generator(sd_list: List[str]) -> Iterable[Tuple[str, str]]:
    for spacer in make_spacer_iter():
        for sd in sd_list:
            yield (sd, spacer)


def run_binding_screening() -> Dict:
    print("\n" + "=" * 80)
    print("STEP 1. Full binding screening for SD random × spacer 9bp")
    print("=" * 80)

    sd_list = make_all_sd_list()
    n_sd = len(sd_list)
    n_spacer = 4 ** SPACER_LEN
    n_total = n_sd * n_spacer

    print(f"SD count              : {n_sd:,}")
    print(f"Spacer length         : {SPACER_LEN} bp")
    print(f"Spacer count          : {n_spacer:,}")
    print(f"Total pairs           : {n_total:,}")
    print(f"Binding target range  : {DG_BIND_MIN} ~ {DG_BIND_MAX}")
    print(f"Output binding file   : {BINDING_CANDIDATES_CSV}")

    nproc = max(1, cpu_count() - 1)
    print(f"Processes             : {nproc}")

    total_done = 0
    n_binding = 0
    dg_sum = 0.0
    dg_min = math.inf
    dg_max = -math.inf

    range_dg_sum = 0.0
    unique_sd_binding = set()
    unique_spacer_binding = set()

    fieldnames = row_fieldnames_base()

    with open(BINDING_CANDIDATES_CSV, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        with Pool(processes=nproc) as pool:
            for row in pool.imap_unordered(binding_worker, task_generator(sd_list), chunksize=2000):
                total_done += 1
                dg = float(row["dg_bind_oasd12__context21"])

                dg_sum += dg
                dg_min = min(dg_min, dg)
                dg_max = max(dg_max, dg)

                if DG_BIND_MIN <= dg <= DG_BIND_MAX:
                    writer.writerow(row)
                    n_binding += 1
                    range_dg_sum += dg
                    unique_sd_binding.add(row["sd_dna"])
                    unique_spacer_binding.add(row["spacer_dna"])

                if total_done % 1_000_000 == 0 or total_done == n_total:
                    mean_dg = dg_sum / total_done
                    ratio = (n_binding / total_done) * 100 if total_done else 0
                    print(
                        f"{total_done:,}/{n_total:,} done | "
                        f"mean_dg_all={mean_dg:.3f} | "
                        f"binding_range={n_binding:,} | "
                        f"ratio={ratio:.6f}%"
                    )

    summary = {
        "n_sd": n_sd,
        "spacer_len": SPACER_LEN,
        "n_spacer": n_spacer,
        "n_total_pairs": n_total,
        "n_evaluated": total_done,
        "mean_dg_all": dg_sum / total_done if total_done else None,
        "min_dg_all": dg_min,
        "max_dg_all": dg_max,
        "n_binding_in_range": n_binding,
        "binding_in_range_ratio_percent": (n_binding / total_done) * 100 if total_done else 0,
        "mean_dg_binding_in_range": range_dg_sum / n_binding if n_binding else None,
        "n_unique_sd_binding_in_range": len(unique_sd_binding),
        "n_unique_spacer_binding_in_range": len(unique_spacer_binding),
        "binding_candidates_csv": BINDING_CANDIDATES_CSV,
    }

    return summary


# ============================================================
# 4) 2차 screening: binding 후보에 직교성 5가지 적용
# ============================================================
def ortho_worker(row: Dict) -> Dict:
    sd_dna = row["sd_dna"]
    spacer_dna = row["spacer_dna"]

    osdsp12_dna = row.get("osdsp12_dna") or build_osdsp12(sd_dna, spacer_dna)
    oasd12_dna = row.get("oasd12_dna") or revcomp_dna(osdsp12_dna)
    oasd6_dna = row.get("oasd6_dna") or revcomp_dna(sd_dna)

    # 조건 1: WT-ASD12 ↔ UP15 + SD + Spacer > 0
    ctx_up15_sd_spacer = UP15_DNA + sd_dna + spacer_dna
    dg_ctx = duplex_dg_dna(WT_ASD_12_DNA, ctx_up15_sd_spacer)
    pass_ctx = dg_ctx > DG_ORTHO_MIN

    # 조건 2: O-ASD12 ↔ WT-SDsp12 > 0
    dg_oasd12_wt_sdsp12 = duplex_dg_dna(oasd12_dna, WT_SDSP12_DNA)
    pass_oasd12_wt_sdsp12 = dg_oasd12_wt_sdsp12 > DG_ORTHO_MIN

    # 조건 3: O-SDsp12 ↔ WT-ASD12 > 0
    dg_osdsp12_wt_asd12 = duplex_dg_dna(osdsp12_dna, WT_ASD_12_DNA)
    pass_osdsp12_wt_asd12 = dg_osdsp12_wt_asd12 > DG_ORTHO_MIN

    # 조건 4: O-SD6 ↔ WT-ASD6 > 0
    dg_osd6_wt_asd6 = duplex_dg_dna(sd_dna, WT_ASD_6_DNA)
    pass_osd6_wt_asd6 = dg_osd6_wt_asd6 > DG_ORTHO_MIN

    # 조건 5: O-ASD6 ↔ WT-SD6 > 0
    dg_oasd6_wt_sd6 = duplex_dg_dna(oasd6_dna, WT_SD_DNA)
    pass_oasd6_wt_sd6 = dg_oasd6_wt_sd6 > DG_ORTHO_MIN

    pass_all = (
        pass_ctx
        and pass_oasd12_wt_sdsp12
        and pass_osdsp12_wt_asd12
        and pass_osd6_wt_asd6
        and pass_oasd6_wt_sd6
    )

    first_fail = "PASS_ALL"
    if not pass_ctx:
        first_fail = "ortho_ctx_wtasd12_up15_sd_spacer"
    elif not pass_oasd12_wt_sdsp12:
        first_fail = "ortho_oasd12_wt_sdsp12"
    elif not pass_osdsp12_wt_asd12:
        first_fail = "ortho_osdsp12_wt_asd12"
    elif not pass_osd6_wt_asd6:
        first_fail = "ortho_osd6_wt_asd6"
    elif not pass_oasd6_wt_sd6:
        first_fail = "ortho_oasd6_wt_sd6"

    out = dict(row)
    out.update({
        "dg_ortho_wtasd12__up15_sd_spacer": dg_ctx,
        "dg_ortho_oasd12__wt_sdsp12": dg_oasd12_wt_sdsp12,
        "dg_ortho_osdsp12__wt_asd12": dg_osdsp12_wt_asd12,
        "dg_ortho_osd6__wt_asd6": dg_osd6_wt_asd6,
        "dg_ortho_oasd6__wt_sd6": dg_oasd6_wt_sd6,
        "pass_ortho_ctx": pass_ctx,
        "pass_ortho_oasd12_wt_sdsp12": pass_oasd12_wt_sdsp12,
        "pass_ortho_osdsp12_wt_asd12": pass_osdsp12_wt_asd12,
        "pass_ortho_osd6_wt_asd6": pass_osd6_wt_asd6,
        "pass_ortho_oasd6_wt_sd6": pass_oasd6_wt_sd6,
        "pass_all": pass_all,
        "first_fail": first_fail,
    })
    return out


def read_binding_rows(path: str) -> Iterable[Dict]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def count_csv_rows(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        # header 제외
        return max(0, sum(1 for _ in f) - 1)


def run_orthogonality_filter(binding_summary: Dict) -> Dict:
    print("\n" + "=" * 80)
    print("STEP 2. Orthogonality filtering for full L9 binding candidates")
    print("=" * 80)

    n_input = count_csv_rows(BINDING_CANDIDATES_CSV)
    print(f"Input binding candidates : {n_input:,}")
    print(f"Input file               : {BINDING_CANDIDATES_CSV}")
    print(f"Output pass_all file     : {PASS_ALL_CSV}")
    print("Criteria                 : all 5 orthogonality ΔG values > 0.0")

    if n_input == 0:
        print("No binding candidates. Stop orthogonality filtering.")
        return {
            "n_input_binding_candidates": 0,
            "n_orthogonality_pass_all": 0,
            "pass_all_ratio_percent": 0,
        }

    nproc = max(1, cpu_count() - 1)

    all_fields = row_fieldnames_base() + ortho_fieldnames_extra()

    n_done = 0
    n_pass = 0
    first_fail_counter = Counter()

    pass_counts = Counter()

    with open(ORTHO_EVALUATED_CSV, "w", newline="", encoding="utf-8") as f_eval, \
         open(PASS_ALL_CSV, "w", newline="", encoding="utf-8") as f_pass:

        eval_writer = csv.DictWriter(f_eval, fieldnames=all_fields)
        pass_writer = csv.DictWriter(f_pass, fieldnames=all_fields)
        eval_writer.writeheader()
        pass_writer.writeheader()

        with Pool(processes=nproc) as pool:
            for out in pool.imap_unordered(ortho_worker, read_binding_rows(BINDING_CANDIDATES_CSV), chunksize=1000):
                n_done += 1
                eval_writer.writerow(out)

                first_fail_counter[out["first_fail"]] += 1

                for col in [
                    "pass_ortho_ctx",
                    "pass_ortho_oasd12_wt_sdsp12",
                    "pass_ortho_osdsp12_wt_asd12",
                    "pass_ortho_osd6_wt_asd6",
                    "pass_ortho_oasd6_wt_sd6",
                ]:
                    if str(out[col]) == "True" or out[col] is True:
                        pass_counts[col] += 1

                if str(out["pass_all"]) == "True" or out["pass_all"] is True:
                    pass_writer.writerow(out)
                    n_pass += 1

                if n_done % 10_000 == 0 or n_done == n_input:
                    ratio = (n_pass / n_done) * 100 if n_done else 0
                    print(
                        f"{n_done:,}/{n_input:,} done | "
                        f"pass_all={n_pass:,} | ratio={ratio:.6f}%"
                    )

    # first_fail summary 저장
    with open(FIRST_FAIL_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["first_fail", "count", "ratio_percent"])
        writer.writeheader()
        for key, count in first_fail_counter.most_common():
            writer.writerow({
                "first_fail": key,
                "count": count,
                "ratio_percent": (count / n_done) * 100 if n_done else 0,
            })

    ortho_summary = {
        "n_input_binding_candidates": n_input,
        "n_orthogonality_evaluated": n_done,
        "n_orthogonality_pass_all": n_pass,
        "pass_all_ratio_percent": (n_pass / n_done) * 100 if n_done else 0,
        "n_pass_ortho_ctx": pass_counts["pass_ortho_ctx"],
        "n_pass_ortho_oasd12_wt_sdsp12": pass_counts["pass_ortho_oasd12_wt_sdsp12"],
        "n_pass_ortho_osdsp12_wt_asd12": pass_counts["pass_ortho_osdsp12_wt_asd12"],
        "n_pass_ortho_osd6_wt_asd6": pass_counts["pass_ortho_osd6_wt_asd6"],
        "n_pass_ortho_oasd6_wt_sd6": pass_counts["pass_ortho_oasd6_wt_sd6"],
        "orthogonality_evaluated_csv": ORTHO_EVALUATED_CSV,
        "pass_all_candidates_csv": PASS_ALL_CSV,
        "first_fail_summary_csv": FIRST_FAIL_CSV,
    }

    return ortho_summary


# ============================================================
# 5) summary 저장
# ============================================================
def save_summary(binding_summary: Dict, ortho_summary: Dict) -> None:
    merged = {}
    merged.update(binding_summary)
    merged.update(ortho_summary)

    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(merged.keys()))
        writer.writeheader()
        writer.writerow(merged)

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Binding candidates     : {merged.get('n_binding_in_range', 0):,}")
    print(f"Orthogonality pass_all : {merged.get('n_orthogonality_pass_all', 0):,}")
    print(f"Pass ratio             : {merged.get('pass_all_ratio_percent', 0):.6f}%")
    print(f"Saved summary          : {SUMMARY_CSV}")
    print(f"Saved binding file     : {BINDING_CANDIDATES_CSV}")
    print(f"Saved pass_all file    : {PASS_ALL_CSV}")


def main():
    binding_summary = run_binding_screening()
    ortho_summary = run_orthogonality_filter(binding_summary)
    save_summary(binding_summary, ortho_summary)


if __name__ == "__main__":
    main()

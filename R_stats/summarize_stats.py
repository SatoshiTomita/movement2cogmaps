#!/usr/bin/env python3
"""Display calculate_stats.r's CSV and save a standalone Japanese HTML table."""

import argparse
import csv
import html
import math
from pathlib import Path


METRICS = {
    "SIr": "SIr：空間情報量",
    "SId": "SId：方向情報量",
    "RVL": "RVL：方向選択性",
    "PlaceCellsPerc": "場所細胞の割合",
    "HDCellsPerc": "頭方向細胞の割合",
    "ConjunctiveCellsPerc": "場所・頭方向の複合選択性細胞の割合",
}
SOURCES = {"model": "モデル", "real": "実測"}
P_COLUMNS = ("jonckheere_p", "wilcox_1_2_p", "wilcox_2_3_p")


def read_rows(path):
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"metric", "source", *P_COLUMNS}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"必要な列がありません: {', '.join(sorted(missing))}")
        for line, row in enumerate(reader, 2):
            values = []
            for column in P_COLUMNS:
                raw = (row.get(column) or "").strip()
                if raw.lower() in {"", "na", "nan", "null"}:
                    values.append(None)
                    continue
                try:
                    value = float(raw)
                except ValueError as exc:
                    raise ValueError(f"{line}行目 {column}: 不正なp値 {raw!r}") from exc
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f"{line}行目 {column}: p値は0〜1で指定してください")
                values.append(value)
            rows.append((
                METRICS.get(row["metric"], row["metric"]),
                SOURCES.get(row["source"], row["source"]),
                values,
            ))
    if not rows:
        raise ValueError("CSVに結果の行がありません")
    return rows


def result_cell(value, alpha):
    if value is None:
        return "欠測", "missing"
    significant = value < alpha
    label = "有意" if significant else "有意ではない"
    return f"{label} (p={value:.6g})", "significant" if significant else "nonsignificant"


def main():
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", type=Path, default=directory / "stats_out.csv")
    parser.add_argument("--output", type=Path, help="HTML保存先（既定: 入力CSVと同じ場所のstats_summary.html）")
    parser.add_argument("--alpha", type=float, default=0.05, help="判定基準（既定: 0.05）")
    parser.add_argument("--groups", nargs=3, default=["群1", "群2", "群3"],
                        metavar=("GROUP1", "GROUP2", "GROUP3"),
                        help="3群の表示名。モデル・実測の両方に適用されます")
    args = parser.parse_args()
    if not 0 < args.alpha < 1:
        parser.error("--alpha は0より大きく1より小さい値にしてください")
    output = args.output or args.csv.with_name("stats_summary.html")
    if output.resolve() == args.csv.resolve():
        parser.error("HTML保存先と入力CSVは別のファイルにしてください")
    try:
        rows = read_rows(args.csv)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    g1, g2, g3 = args.groups
    headers = ["指標", "対象", "JT：全体の増加傾向",
               f"Wilcoxon：{g2} > {g1}", f"Wilcoxon：{g3} > {g2}"]
    notes = [
        f"判定はp < {args.alpha:g}。有意ではない結果は「差がない」ことの証明ではありません。",
        "calculate_stats.rの設定：JTは増加方向・10,000回の置換、Wilcoxonは片側・対応なし。",
        "Wilcoxonのp値は各指標・各対象内の群間比較でBH補正済み。JTのp値は未補正です。追加の補正は行っていません。",
        "p値は変化の大きさを表しません。JTのp=0.0001付近は置換回数による分解能に注意してください。",
        "この表にはJS距離・モデルと実測の直接比較・seed間の再現性検定は含まれません。",
        "入力CSVにはseedや群の意味の記録がありません。解析対象と群番号の対応は入力データの生成処理で確認してください。",
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    html_rows = []
    for metric, source, values in rows:
        cells = [result_cell(value, args.alpha) for value in values]
        print("| " + " | ".join([metric, source, *(label for label, _ in cells)]) + " |")
        html_rows.append(
            "<tr><th scope='row'>" + html.escape(metric) + "</th><td>" + html.escape(source)
            + "</td>" + "".join(
                f"<td class='{kind}'>{html.escape(label)}</td>" for label, kind in cells
            ) + "</tr>"
        )
    document = """<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>統計検定の結果</title>
<style>
body {font-family:system-ui,sans-serif;max-width:1200px;margin:32px auto;padding:0 20px;color:#172b3a;line-height:1.6}
h1 {font-size:1.6rem} .table {overflow-x:auto}
table {border-collapse:collapse;width:100%;font-size:.95rem}
th,td {padding:12px;border:1px solid #cbd5df;text-align:left}
thead {background:#edf2f7} td {white-space:nowrap}
.significant {background:#e8f4ee;color:#16543a}
.nonsignificant {background:#f5f5f5;color:#424242}
.missing {color:#666} li {margin:8px 0}
@media print {body {margin:0;padding:0} th,td {padding:6px} .table {overflow:visible}}
</style></head><body><h1>統計検定の結果</h1>"""
    document += "<p>入力：" + html.escape(str(args.csv.resolve())) + "</p>"
    document += "<div class='table'><table><thead><tr>" + "".join(
        "<th scope='col'>" + html.escape(header) + "</th>" for header in headers
    ) + "</tr></thead><tbody>" + "".join(html_rows) + "</tbody></table></div>"
    document += "<ul>" + "".join("<li>" + html.escape(note) + "</li>" for note in notes)
    document += "</ul></body></html>\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    print("\n" + "\n".join(notes))
    print(f"\nHTML保存先: {output.resolve()}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from io import BytesIO, StringIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
from docx import Document


def _flatten_dict(data: dict, parent_key: str = "", sep: str = ".") -> dict:
    items: dict[str, object] = {}
    for key, value in data.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else str(key)
        if isinstance(value, dict):
            items.update(_flatten_dict(value, parent_key=new_key, sep=sep))
        elif isinstance(value, list):
            items[new_key] = json.dumps(value, ensure_ascii=False)
        else:
            items[new_key] = value
    return items


def _normalize_cell(value: object) -> object:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def build_preview_tables(data: dict) -> dict:
    """Build a canonical, human-readable table pack with predictable
    Field/Value layouts, plus dedicated table sheets for tabular content.
    This prevents distorted exports caused by over-flattening.
    """
    tables: dict[str, pd.DataFrame] = {}

    # 1) Core metadata summary (scalar-only)
    scalar_rows = []
    for key, value in data.items():
        if isinstance(value, (dict, list)):
            continue
        scalar_rows.append({"Field": str(key), "Value": _normalize_cell(value)})
    if scalar_rows:
        tables["Summary"] = pd.DataFrame(scalar_rows)

    # 2) Strongly typed field-value blocks
    fields = data.get("fields")
    if isinstance(fields, dict):
        rows = [{"Field": str(k), "Value": _normalize_cell(v)} for k, v in fields.items()]
        if rows:
            tables["Fields"] = pd.DataFrame(rows)

    raw_kv = data.get("raw_key_values")
    if isinstance(raw_kv, dict):
        rows = [{"Field": str(k), "Value": _normalize_cell(v)} for k, v in raw_kv.items()]
        if rows:
            tables["RawKeyValues"] = pd.DataFrame(rows)

    metrics = data.get("metrics")
    if isinstance(metrics, dict):
        rows = [{"Field": str(k), "Value": _normalize_cell(v)} for k, v in metrics.items()]
        if rows:
            tables["Metrics"] = pd.DataFrame(rows)

    # 3) Text excerpts in row form (page/value) to avoid giant-cell distortion
    native_excerpt = data.get("native_text_excerpt")
    if isinstance(native_excerpt, dict):
        rows = [{"Page": str(k), "Text": str(v)} for k, v in native_excerpt.items()]
        if rows:
            tables["NativeTextExcerpt"] = pd.DataFrame(rows)

    ocr_excerpt = data.get("ocr_text_excerpt")
    if isinstance(ocr_excerpt, dict):
        rows = [{"Page": str(k), "Text": str(v)} for k, v in ocr_excerpt.items()]
        if rows:
            tables["OCRTextExcerpt"] = pd.DataFrame(rows)

    # 4) Extracted nested tables preserved as true tables
    nested_tables = data.get("tables")
    if isinstance(nested_tables, dict):
        for name, rows in nested_tables.items():
            if isinstance(rows, list) and rows and all(isinstance(r, dict) for r in rows):
                df = pd.DataFrame(rows)
                if not df.empty:
                    tables[f"Table_{str(name)}"[:31]] = df

    # 5) Any remaining dict/list content represented cleanly
    for key, value in data.items():
        if key in (
            "fields",
            "raw_key_values",
            "metrics",
            "native_text_excerpt",
            "ocr_text_excerpt",
            "tables",
        ):
            continue

        if isinstance(value, dict):
            rows = [{"Field": str(k), "Value": _normalize_cell(v)} for k, v in value.items()]
            if rows:
                tables[f"{str(key)[:31]}"] = pd.DataFrame(rows)

        elif isinstance(value, list):
            if value and all(isinstance(item, dict) for item in value):
                df = pd.DataFrame(value)
                if not df.empty:
                    tables[f"{str(key)[:31]}"] = df
            elif value:
                rows = [{"Index": i + 1, "Value": _normalize_cell(v)} for i, v in enumerate(value)]
                tables[f"{str(key)[:31]}"] = pd.DataFrame(rows)

    if not tables:
        tables["Summary"] = pd.DataFrame(
            [{"Field": "data", "Value": json.dumps(data, ensure_ascii=False)}]
        )

    return tables


def export_json_bytes(data: dict) -> bytes:
    return json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")


def export_csv_bytes(data: dict) -> bytes:
    """Single CSV export with section markers and clean row/column tables."""
    tables = build_preview_tables(data)
    buffer = StringIO()

    first = True
    for name, df in tables.items():
        if not first:
            buffer.write("\n")
        first = False
        buffer.write(f"### {name} ###\n")
        clean_df = df.copy()
        for col in clean_df.columns:
            clean_df[col] = clean_df[col].map(_normalize_cell)
        clean_df.to_csv(buffer, index=False)

    return buffer.getvalue().encode("utf-8")


def export_excel_bytes(data: dict) -> bytes:
    output = BytesIO()
    tables = build_preview_tables(data)

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        used_sheet_names: set[str] = set()

        for raw_name, df in tables.items():
            base = str(raw_name)[:31] or "Sheet"
            sheet_name = base
            counter = 1
            while sheet_name in used_sheet_names:
                suffix = f"_{counter}"
                sheet_name = f"{base[:31-len(suffix)]}{suffix}"
                counter += 1
            used_sheet_names.add(sheet_name)

            clean_df = df.copy()
            for col in clean_df.columns:
                clean_df[col] = clean_df[col].map(_normalize_cell)
            clean_df.to_excel(writer, sheet_name=sheet_name, index=False)

        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            worksheet.freeze_panes = "A2"

            header = next(worksheet.iter_rows(min_row=1, max_row=1))
            for cell in header:
                cell.font = cell.font.copy(bold=True)

            for column_cells in worksheet.columns:
                max_length = max(
                    (len(str(cell.value)) for cell in column_cells if cell.value is not None),
                    default=10,
                )
                col_letter = column_cells[0].column_letter
                worksheet.column_dimensions[col_letter].width = min(max(max_length + 2, 12), 80)

    return output.getvalue()


def export_word_bytes(data: dict, title: str = "Document Report") -> bytes:
    doc = Document()
    doc.add_heading(title, level=1)

    tables = build_preview_tables(data)

    for name, df in tables.items():
        doc.add_heading(name, level=2)
        if df.empty:
            continue

        clean_df = df.copy()
        for col in clean_df.columns:
            clean_df[col] = clean_df[col].map(_normalize_cell)

        headers = list(clean_df.columns)
        table = doc.add_table(rows=1, cols=len(headers))
        table.style = "Table Grid"

        for idx, header in enumerate(headers):
            table.rows[0].cells[idx].text = str(header)

        for _, row in clean_df.iterrows():
            cells = table.add_row().cells
            for idx, header in enumerate(headers):
                value = row[header]
                cells[idx].text = str(value) if pd.notna(value) else ""

    out = BytesIO()
    doc.save(out)
    return out.getvalue()


def export_txt_bytes(text_by_page: dict[int, str]) -> bytes:
    lines = []
    for page, content in text_by_page.items():
        lines.append(f"[Page {page}]")
        lines.append(content)
        lines.append("")
    return "\n".join(lines).encode("utf-8")


def export_zip_bytes(file_map: dict[str, bytes]) -> bytes:
    out = BytesIO()
    with ZipFile(out, "w", compression=ZIP_DEFLATED) as zip_file:
        for filename, content in file_map.items():
            zip_file.writestr(filename, content)
    return out.getvalue()


def build_export_package(base_name: str, structured_data: dict, native_text_by_page: dict[int, str], api_payload: dict) -> dict[str, bytes]:
    safe_name = Path(base_name).stem or "document"

    json_bytes = export_json_bytes(structured_data)
    csv_bytes = export_csv_bytes(structured_data)
    xlsx_bytes = export_excel_bytes(structured_data)
    docx_bytes = export_word_bytes(structured_data, title=f"{safe_name} Structured Report")
    txt_bytes = export_txt_bytes(native_text_by_page)
    api_json_bytes = export_json_bytes(api_payload)

    package = {
        f"{safe_name}.structured.json": json_bytes,
        f"{safe_name}.structured.csv": csv_bytes,
        f"{safe_name}.structured.xlsx": xlsx_bytes,
        f"{safe_name}.structured.docx": docx_bytes,
        f"{safe_name}.raw_text.txt": txt_bytes,
        f"{safe_name}.api_payload.json": api_json_bytes,
    }

    package[f"{safe_name}.bundle.zip"] = export_zip_bytes(package)
    return package

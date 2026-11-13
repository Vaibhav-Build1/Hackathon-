import json
from pathlib import Path

import pandas as pd
from docx import Document


class BasePublisher:
    name = "base"

    def publish(self, structured_data: dict, output_dir: Path, source_pdf: Path) -> Path:
        raise NotImplementedError


class JSONPublisher(BasePublisher):
    name = "json"

    def publish(self, structured_data: dict, output_dir: Path, source_pdf: Path) -> Path:
        path = output_dir / f"{source_pdf.stem}.structured.json"
        path.write_text(json.dumps(structured_data, indent=2), encoding="utf-8")
        return path


class CSVPublisher(BasePublisher):
    name = "csv"

    def publish(self, structured_data: dict, output_dir: Path, source_pdf: Path) -> Path:
        path = output_dir / f"{source_pdf.stem}.fields.csv"
        fields = structured_data.get("fields", {})
        df = pd.DataFrame([fields]) if fields else pd.DataFrame([structured_data.get("raw_key_values", {})])
        df.to_csv(path, index=False)
        return path


class ExcelPublisher(BasePublisher):
    name = "excel"

    def publish(self, structured_data: dict, output_dir: Path, source_pdf: Path) -> Path:
        path = output_dir / f"{source_pdf.stem}.xlsx"
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            fields = structured_data.get("fields", {})
            pd.DataFrame([fields]).to_excel(writer, sheet_name="fields", index=False)
            tables = structured_data.get("tables", {})
            for sheet, rows in tables.items():
                pd.DataFrame(rows).to_excel(writer, sheet_name=sheet[:31], index=False)
        return path


class WordPublisher(BasePublisher):
    name = "word"

    def publish(self, structured_data: dict, output_dir: Path, source_pdf: Path) -> Path:
        path = output_dir / f"{source_pdf.stem}.docx"
        doc = Document()
        doc.add_heading("Extracted Structured Data", level=1)
        doc.add_paragraph(f"Document Type: {structured_data.get('document_type', 'unknown')}")

        doc.add_heading("Fields", level=2)
        for key, value in structured_data.get("fields", {}).items():
            doc.add_paragraph(f"{key}: {value}")

        doc.add_heading("Raw Key/Value", level=2)
        for key, value in structured_data.get("raw_key_values", {}).items():
            doc.add_paragraph(f"{key}: {value}")

        tables = structured_data.get("tables", {})
        for table_name, rows in tables.items():
            doc.add_heading(table_name, level=2)
            if not rows:
                continue
            headers = list(rows[0].keys())
            t = doc.add_table(rows=1, cols=len(headers))
            for i, header in enumerate(headers):
                t.rows[0].cells[i].text = header
            for row in rows:
                cells = t.add_row().cells
                for i, header in enumerate(headers):
                    cells[i].text = str(row.get(header, ""))

        doc.save(path)
        return path


class ConcurMockPublisher(BasePublisher):
    name = "concur"

    def publish(self, structured_data: dict, output_dir: Path, source_pdf: Path) -> Path:
        payload = {
            "action": "create_or_match_expense",
            "source_pdf": str(source_pdf),
            "expense": structured_data.get("fields", {}),
            "attachment": source_pdf.name,
        }
        path = output_dir / f"{source_pdf.stem}.concur.mock.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path


class ResumeTemplatePublisher(BasePublisher):
    name = "resume_template"

    def publish(self, structured_data: dict, output_dir: Path, source_pdf: Path) -> Path:
        path = output_dir / f"{source_pdf.stem}.resume.template.json"
        fields = structured_data.get("fields", {})
        payload = {
            "template": "standard_resume_template",
            "candidate": {
                "name": fields.get("candidate_name", ""),
                "email": fields.get("email", ""),
                "phone": fields.get("phone", ""),
                "linkedin": fields.get("linkedin", ""),
            },
            "raw": structured_data,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path


PUBLISHER_REGISTRY = {
    p.name: p
    for p in [
        JSONPublisher(),
        CSVPublisher(),
        ExcelPublisher(),
        WordPublisher(),
        ConcurMockPublisher(),
        ResumeTemplatePublisher(),
    ]
}


def available_publishers() -> list[str]:
    return sorted(PUBLISHER_REGISTRY.keys())


def publish(structured_data: dict, output_dir: Path, targets: list[str], source_pdf: Path) -> list[dict]:
    published = []
    for target in targets:
        publisher = PUBLISHER_REGISTRY.get(target)
        if not publisher:
            continue
        output_path = publisher.publish(structured_data=structured_data, output_dir=output_dir, source_pdf=source_pdf)
        published.append({"target": target, "path": str(output_path)})
    return published

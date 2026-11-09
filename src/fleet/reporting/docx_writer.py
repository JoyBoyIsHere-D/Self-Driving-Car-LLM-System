"""A tiny .docx writer (no third-party packages): headings, paragraphs with
**bold** markup, bullet/numbered lists, tables, images, captions, code
blocks, call-out boxes, page breaks and an auto-updating table of contents."""
from __future__ import annotations

import os
import re
import struct
import zipfile
from xml.sax.saxutils import escape

EMU_PER_INCH = 914400
ACCENT = "2A78D6"


def _png_size(path: str) -> tuple[int, int]:
    with open(path, "rb") as f:
        head = f.read(24)
    w, h = struct.unpack(">II", head[16:24])
    return w, h


def _runs(text: str, size: float | None = None, color: str | None = None, italic: bool = False,
          font: str | None = None, bold_all: bool = False) -> str:
    """Turn 'plain **bold** plain `code`' into Word runs."""
    out = []
    for part in re.split(r"(\*\*[^*]+\*\*|`[^`]+`)", text):
        if not part:
            continue
        bold, mono = bold_all, font
        if part.startswith("**") and part.endswith("**"):
            part, bold = part[2:-2], True
        elif part.startswith("`") and part.endswith("`"):
            part, mono = part[1:-1], "Consolas"
        rpr = ""
        if mono:
            rpr += f'<w:rFonts w:ascii="{mono}" w:hAnsi="{mono}" w:cs="{mono}"/>'
        if bold:
            rpr += "<w:b/>"
        if italic:
            rpr += "<w:i/>"
        if color:
            rpr += f'<w:color w:val="{color}"/>'
        if size:
            rpr += f'<w:sz w:val="{int(size * 2)}"/>'
        out.append(f'<w:r><w:rPr>{rpr}</w:rPr><w:t xml:space="preserve">{escape(part)}</w:t></w:r>')
    return "".join(out)


class Docx:
    def __init__(self):
        self.body: list[str] = []
        self.media: list[tuple[str, str]] = []   # (rId, source path)
        self._fig = 0
        self._tab = 0
        self._pic_id = 1

    # ------------------------------------------------------------- blocks
    def heading(self, text: str, level: int = 1) -> None:
        self.body.append(f'<w:p><w:pPr><w:pStyle w:val="Heading{level}"/></w:pPr>{_runs(text)}</w:p>')

    def title(self, text: str, subtitle: str = "") -> None:
        self.body.append(f'<w:p><w:pPr><w:pStyle w:val="Title"/></w:pPr>{_runs(text)}</w:p>')
        if subtitle:
            self.body.append(f'<w:p><w:pPr><w:pStyle w:val="Subtitle"/></w:pPr>{_runs(subtitle)}</w:p>')

    def para(self, text: str, align: str | None = None, size: float | None = None, color: str | None = None,
             italic: bool = False, bold: bool = False, space_after: int | None = None) -> None:
        ppr = ""
        if align:
            ppr += f'<w:jc w:val="{align}"/>'
        if space_after is not None:
            ppr += f'<w:spacing w:after="{space_after}"/>'
        self.body.append(f"<w:p><w:pPr>{ppr}</w:pPr>{_runs(text, size, color, italic, bold_all=bold)}</w:p>")

    def bullets(self, items: list[str], numbered: bool = False) -> None:
        num = 2 if numbered else 1
        for it in items:
            level = 0
            while it.startswith("  "):
                it, level = it[2:], level + 1
            self.body.append(f'<w:p><w:pPr><w:pStyle w:val="ListParagraph"/><w:numPr><w:ilvl w:val="{level}"/>'
                             f'<w:numId w:val="{num}"/></w:numPr><w:spacing w:after="60"/></w:pPr>{_runs(it)}</w:p>')
        if numbered:
            self._restart_numbering()

    def _restart_numbering(self) -> None:
        # each numbered list gets a fresh numId so it starts at 1 again
        self._num_restarts = getattr(self, "_num_restarts", 0) + 1
        new_id = 10 + self._num_restarts
        self.body = [b.replace('<w:numId w:val="2"/>', f'<w:numId w:val="{new_id}"/>') for b in self.body]

    def code(self, text: str) -> None:
        lines = text.rstrip("\n").split("\n")
        for i, line in enumerate(lines):
            after = 200 if i == len(lines) - 1 else 0
            self.body.append(f'<w:p><w:pPr><w:pStyle w:val="Code"/><w:spacing w:after="{after}" w:line="240" '
                             f'w:lineRule="auto"/></w:pPr><w:r><w:t xml:space="preserve">{escape(line)}</w:t></w:r></w:p>')

    def callout(self, text: str, title: str | None = None) -> None:
        inner = (_runs(title + "  ", bold_all=True, color=ACCENT) if title else "") + _runs(text)
        self.body.append('<w:p><w:pPr><w:pBdr><w:left w:val="single" w:sz="24" w:space="8" w:color="2A78D6"/></w:pBdr>'
                         '<w:shd w:val="clear" w:color="auto" w:fill="EEF4FC"/><w:spacing w:before="120" w:after="200"/>'
                         f'<w:ind w:left="180" w:right="180"/></w:pPr>{inner}</w:p>')

    def table(self, rows: list[list[str]], widths: list[float] | None = None, caption: str | None = None) -> None:
        ncol = len(rows[0])
        widths = widths or [6.5 / ncol] * ncol
        tw = [int(w * 1440) for w in widths]
        grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in tw)
        xml = ['<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/><w:tblW w:w="0" w:type="auto"/>'
               '<w:tblLayout w:type="fixed"/><w:tblCellMar><w:top w:w="50" w:type="dxa"/><w:left w:w="90" w:type="dxa"/>'
               '<w:bottom w:w="50" w:type="dxa"/><w:right w:w="90" w:type="dxa"/></w:tblCellMar></w:tblPr>'
               f'<w:tblGrid>{grid}</w:tblGrid>']
        for r, row in enumerate(rows):
            head = r == 0
            trpr = "<w:trPr><w:tblHeader/><w:cantSplit/></w:trPr>" if head else "<w:trPr><w:cantSplit/></w:trPr>"
            cells = []
            for c, val in enumerate(row):
                shade = '<w:shd w:val="clear" w:color="auto" w:fill="E8F0FB"/>' if head else ""
                cells.append(f'<w:tc><w:tcPr><w:tcW w:w="{tw[c]}" w:type="dxa"/>{shade}</w:tcPr>'
                             f'<w:p><w:pPr><w:spacing w:after="0"/></w:pPr>{_runs(str(val), size=9.5, bold_all=head)}</w:p></w:tc>')
            xml.append(f"<w:tr>{trpr}{''.join(cells)}</w:tr>")
        xml.append("</w:tbl>")
        if caption:
            self._tab += 1
            self.body.append(f'<w:p><w:pPr><w:pStyle w:val="Caption"/><w:keepNext/></w:pPr>'
                             f'{_runs(f"Table {self._tab}. {caption}")}</w:p>')
        self.body.append("".join(xml))
        self.body.append('<w:p><w:pPr><w:spacing w:after="80"/></w:pPr></w:p>')

    def image(self, path: str, width_in: float = 6.5, caption: str | None = None) -> None:
        if not os.path.exists(path):
            self.para(f"[missing figure: {path}]", italic=True, color="999999")
            return
        rid = f"rIdImg{len(self.media) + 1}"
        self.media.append((rid, path))
        w, h = _png_size(path)
        cx = int(width_in * EMU_PER_INCH)
        cy = int(cx * h / w)
        pid = self._pic_id
        self._pic_id += 1
        name = escape(os.path.basename(path))
        self.body.append(
            '<w:p><w:pPr><w:jc w:val="center"/><w:keepNext/><w:spacing w:after="60"/></w:pPr><w:r><w:drawing>'
            f'<wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/>'
            f'<wp:docPr id="{pid}" name="{name}"/><wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/>'
            '</wp:cNvGraphicFramePr><a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            f'<pic:pic><pic:nvPicPr><pic:cNvPr id="{pid}" name="{name}"/><pic:cNvPicPr/></pic:nvPicPr>'
            f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
            f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
            '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic>'
            '</wp:inline></w:drawing></w:r></w:p>')
        if caption:
            self._fig += 1
            self.body.append(f'<w:p><w:pPr><w:pStyle w:val="Caption"/></w:pPr>{_runs(f"Figure {self._fig}. {caption}")}</w:p>')

    def page_break(self) -> None:
        self.body.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')

    def toc(self) -> None:
        self.body.append('<w:p><w:pPr><w:pStyle w:val="TOCHeading"/></w:pPr><w:r><w:t>Contents</w:t></w:r></w:p>')
        self.body.append('<w:p><w:r><w:fldChar w:fldCharType="begin" w:dirty="true"/></w:r><w:r><w:instrText xml:space="preserve">'
                         ' TOC \\o "1-2" \\h \\z \\u </w:instrText></w:r><w:r><w:fldChar w:fldCharType="separate"/></w:r>'
                         '<w:r><w:rPr><w:i/><w:color w:val="777777"/></w:rPr><w:t>The table of contents fills in when Word '
                         'opens the file (click "Yes" to update fields, or right-click here and choose Update Field).</w:t></w:r>'
                         '<w:r><w:fldChar w:fldCharType="end"/></w:r></w:p>')

    # ------------------------------------------------------------- save
    def save(self, path: str, title: str = "Report", author: str = "") -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        body = "".join(self.body)
        doc = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
               '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
               'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
               'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
               'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
               'xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
               f'<w:body>{body}<w:sectPr><w:footerReference w:type="default" r:id="rIdFooter"/>'
               '<w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1300" w:right="1300" w:bottom="1300" '
               'w:left="1300" w:header="600" w:footer="600" w:gutter="0"/></w:sectPr></w:body></w:document>')
        rels = ['<Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>',
                '<Relationship Id="rIdNum" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/numbering" Target="numbering.xml"/>',
                '<Relationship Id="rIdSettings" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings" Target="settings.xml"/>',
                '<Relationship Id="rIdFooter" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" Target="footer1.xml"/>']
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            for i, (rid, src) in enumerate(self.media):
                target = f"media/image{i + 1}.png"
                z.write(src, f"word/{target}")
                rels.append(f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="{target}"/>')
            z.writestr("[Content_Types].xml", CONTENT_TYPES)
            z.writestr("_rels/.rels", ROOT_RELS)
            z.writestr("docProps/core.xml", CORE.format(title=escape(title), author=escape(author)))
            z.writestr("docProps/app.xml", APP)
            z.writestr("word/document.xml", doc)
            z.writestr("word/styles.xml", STYLES)
            z.writestr("word/numbering.xml", self._numbering())
            z.writestr("word/settings.xml", SETTINGS)
            z.writestr("word/footer1.xml", FOOTER)
            z.writestr("word/_rels/document.xml.rels",
                       '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships '
                       'xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' + "".join(rels) + "</Relationships>")

    def _numbering(self) -> str:
        bullet_lvls = "".join(
            f'<w:lvl w:ilvl="{i}"><w:start w:val="1"/><w:numFmt w:val="bullet"/><w:lvlText w:val="{c}"/>'
            f'<w:lvlJc w:val="left"/><w:pPr><w:ind w:left="{360 + 360 * i}" w:hanging="260"/></w:pPr>'
            f'<w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/><w:color w:val="2A78D6"/></w:rPr></w:lvl>'
            for i, c in enumerate(["●", "–", "·"]))
        num_lvls = "".join(
            f'<w:lvl w:ilvl="{i}"><w:start w:val="1"/><w:numFmt w:val="{f}"/><w:lvlText w:val="%{i + 1}."/>'
            f'<w:lvlJc w:val="left"/><w:pPr><w:ind w:left="{400 + 360 * i}" w:hanging="320"/></w:pPr></w:lvl>'
            for i, f in enumerate(["decimal", "lowerLetter", "lowerRoman"]))
        nums = '<w:num w:numId="1"><w:abstractNumId w:val="0"/></w:num><w:num w:numId="2"><w:abstractNumId w:val="1"/></w:num>'
        for k in range(1, getattr(self, "_num_restarts", 0) + 1):
            nums += (f'<w:num w:numId="{10 + k}"><w:abstractNumId w:val="1"/><w:lvlOverride w:ilvl="0">'
                     '<w:startOverride w:val="1"/></w:lvlOverride></w:num>')
        return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<w:numbering xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                f'<w:abstractNum w:abstractNumId="0"><w:multiLevelType w:val="hybridMultilevel"/>{bullet_lvls}</w:abstractNum>'
                f'<w:abstractNum w:abstractNumId="1"><w:multiLevelType w:val="hybridMultilevel"/>{num_lvls}</w:abstractNum>'
                f'{nums}</w:numbering>')


CONTENT_TYPES = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                 '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                 '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                 '<Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/>'
                 '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
                 '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
                 '<Override PartName="/word/numbering.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml"/>'
                 '<Override PartName="/word/settings.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml"/>'
                 '<Override PartName="/word/footer1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/>'
                 '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
                 '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
                 '</Types>')
ROOT_RELS = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
             '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
             '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
             '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
             '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>'
             '</Relationships>')
CORE = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"><dc:title>{title}</dc:title><dc:creator>{author}</dc:creator>'
        '</cp:coreProperties>')
APP = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
       '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"><Application>fleet report builder</Application></Properties>')
SETTINGS = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:settings xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            '<w:updateFields w:val="true"/><w:defaultTabStop w:val="720"/></w:settings>')
FOOTER = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
          '<w:ftr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:p><w:pPr><w:jc w:val="center"/></w:pPr>'
          '<w:r><w:rPr><w:color w:val="888888"/><w:sz w:val="18"/></w:rPr><w:fldChar w:fldCharType="begin"/></w:r>'
          '<w:r><w:rPr><w:color w:val="888888"/><w:sz w:val="18"/></w:rPr><w:instrText>PAGE</w:instrText></w:r>'
          '<w:r><w:rPr><w:color w:val="888888"/><w:sz w:val="18"/></w:rPr><w:fldChar w:fldCharType="end"/></w:r></w:p></w:ftr>')


def _style(sid, name, ppr="", rpr="", based="Normal", next_="Normal", extra=""):
    return (f'<w:style w:type="paragraph" w:styleId="{sid}"><w:name w:val="{name}"/><w:basedOn w:val="{based}"/>'
            f'<w:next w:val="{next_}"/>{extra}<w:qFormat/><w:pPr>{ppr}</w:pPr><w:rPr>{rpr}</w:rPr></w:style>')


STYLES = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
          '<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
          '<w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:eastAsia="Calibri" w:cs="Calibri"/>'
          '<w:sz w:val="22"/><w:szCs w:val="22"/><w:color w:val="1F1F1F"/></w:rPr></w:rPrDefault>'
          '<w:pPrDefault><w:pPr><w:spacing w:after="140" w:line="276" w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults>'
          '<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:qFormat/></w:style>'
          + _style("Title", "Title", '<w:spacing w:before="1800" w:after="200"/>',
                   '<w:b/><w:sz w:val="52"/><w:color w:val="0B0B0B"/>')
          + _style("Subtitle", "Subtitle", '<w:spacing w:after="600"/>', '<w:sz w:val="28"/><w:color w:val="52514E"/>')
          + _style("Heading1", "heading 1", '<w:keepNext/><w:spacing w:before="360" w:after="160"/><w:outlineLvl w:val="0"/>'
                   '<w:pBdr><w:bottom w:val="single" w:sz="8" w:space="4" w:color="2A78D6"/></w:pBdr>',
                   '<w:b/><w:sz w:val="34"/><w:color w:val="0B0B0B"/>')
          + _style("Heading2", "heading 2", '<w:keepNext/><w:spacing w:before="280" w:after="100"/><w:outlineLvl w:val="1"/>',
                   '<w:b/><w:sz w:val="27"/><w:color w:val="1F5FA8"/>')
          + _style("Heading3", "heading 3", '<w:keepNext/><w:spacing w:before="200" w:after="80"/><w:outlineLvl w:val="2"/>',
                   '<w:b/><w:sz w:val="23"/><w:color w:val="333333"/>')
          + _style("Caption", "caption", '<w:spacing w:before="40" w:after="240"/><w:jc w:val="center"/>',
                   '<w:i/><w:sz w:val="19"/><w:color w:val="52514E"/>')
          + _style("Code", "Code", '<w:shd w:val="clear" w:color="auto" w:fill="F3F3F1"/><w:ind w:left="200" w:right="200"/>',
                   '<w:rFonts w:ascii="Consolas" w:hAnsi="Consolas" w:cs="Consolas"/><w:sz w:val="18"/>')
          + _style("ListParagraph", "List Paragraph", '<w:ind w:left="720"/>')
          + _style("TOCHeading", "TOC Heading", '<w:spacing w:before="240" w:after="120"/>', '<w:b/><w:sz w:val="32"/>')
          + _style("TOC1", "toc 1", '<w:spacing w:after="60"/><w:tabs><w:tab w:val="right" w:leader="dot" w:pos="9300"/></w:tabs>', '<w:b/>')
          + _style("TOC2", "toc 2", '<w:spacing w:after="40"/><w:ind w:left="300"/><w:tabs><w:tab w:val="right" w:leader="dot" w:pos="9300"/></w:tabs>')
          + '<w:style w:type="table" w:styleId="TableGrid"><w:name w:val="Table Grid"/><w:tblPr><w:tblBorders>'
          '<w:top w:val="single" w:sz="4" w:color="C9C8C3"/><w:left w:val="single" w:sz="4" w:color="C9C8C3"/>'
          '<w:bottom w:val="single" w:sz="4" w:color="C9C8C3"/><w:right w:val="single" w:sz="4" w:color="C9C8C3"/>'
          '<w:insideH w:val="single" w:sz="4" w:color="C9C8C3"/><w:insideV w:val="single" w:sz="4" w:color="C9C8C3"/>'
          '</w:tblBorders></w:tblPr></w:style>'
          '</w:styles>')

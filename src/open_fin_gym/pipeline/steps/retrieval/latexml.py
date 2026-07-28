import re

from lxml import html
from lxml.html import HtmlElement

# LaTeXML titles map onto the heading levels the chunker splits on. Run-in titles
# (paragraph, theorem, proof) are rendered as h5/h6 but are body text, not sections.
HEADING_LEVELS = {
    "ltx_title_document": "#",
    "ltx_title_abstract": "##",
    "ltx_title_section": "##",
    "ltx_title_bibliography": "##",
    "ltx_title_appendix": "##",
    "ltx_title_subsection": "###",
    "ltx_title_subsubsection": "####",
}
BLOCK_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "figcaption", "pre")
DROP_TAGS = ("script", "style", "nav")
WHITESPACE = re.compile(r"\s+")


def arxiv_html_to_markdown(source: bytes) -> str:
    """
    Convert an arXiv LaTeXML-rendered HTML paper into markdown

    Tables and maths are recovered from the rendered LaTeX rather than from page
    layout, so numeric result tables and equations survive intact.

    Args:
        source: Raw HTML page body

    Returns:
        Markdown document with ATX headers
    """
    root = html.fromstring(source)

    for element in list(root.iter(*DROP_TAGS)):
        element.getparent().remove(element)

    # Inline the LaTeX maths before any text is read out of the tree
    for element in list(root.iter("math")):
        _replace_with_text(element, f"${element.get('alttext') or _text(element)}$")

    for element in _tables(root):
        _replace_with_text(element, _table_to_markdown(element), tag="pre")

    blocks = []
    for element in root.iter(*BLOCK_TAGS):
        if element.tag == "pre":
            if element.text:
                blocks.append(element.text)
            continue

        # A tabular is wrapped in a paragraph, which would flatten it if read here
        if element.find(".//pre") is not None:
            continue

        text = _text(element)
        if not text:
            continue

        level = HEADING_LEVELS.get(_title_class(element))
        if level:
            blocks.append(f"{level} {text}")
        elif element.tag == "li":
            blocks.append(f"- {text}")
        else:
            blocks.append(text)

    return "\n\n".join(blocks)


def _tables(root: HtmlElement) -> list[HtmlElement]:
    """
    Find the outermost LaTeXML tabulars

    LaTeXML emits a tabular as either a `table` or, when it appears inside an
    inline context such as `resizebox`, a `span`. Nested tabulars are skipped as
    the enclosing one already renders them.

    Args:
        root: Document root

    Returns:
        List of tabular elements in document order
    """
    tabulars = root.find_class("ltx_tabular")
    return [x for x in tabulars if not any(y in tabulars for y in x.iterancestors())]


def _table_to_markdown(table: HtmlElement) -> str:
    """
    Render a LaTeXML tabular as a markdown table

    Args:
        table: Tabular element

    Returns:
        Markdown table, using the first row as the header
    """
    rows = []
    for row in table.find_class("ltx_tr"):
        cells = []
        for cell in row.find_class("ltx_td"):
            cells.append(_text(cell).replace("|", "\\|"))
            # A spanning cell occupies the columns it covers so the grid stays square
            cells.extend([""] * (int(cell.get("colspan", 1)) - 1))
        rows.append(cells)

    if not rows:
        return ""

    width = max(len(x) for x in rows)
    rows = [x + [""] * (width - len(x)) for x in rows]
    header = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    return "\n".join(header + ["| " + " | ".join(x) + " |" for x in rows[1:]])


def _replace_with_text(element: HtmlElement, text: str, tag: str = "span") -> None:
    """
    Replace an element's subtree with a text node, preserving its position

    Args:
        element: Element to collapse
        text: Replacement text
        tag: Tag to relabel the element as
    """
    for child in list(element):
        element.remove(child)
    element.tag = tag
    element.text = text


def _title_class(element: HtmlElement) -> str | None:
    """
    Get the LaTeXML title class of an element, e.g. 'ltx_title_section'

    Args:
        element: Candidate heading element

    Returns:
        Title class, or None if the element is not a LaTeXML title
    """
    return next(
        (x for x in element.get("class", "").split() if x in HEADING_LEVELS), None
    )


def _text(element: HtmlElement) -> str:
    """
    Get the whitespace-normalised text of an element

    Args:
        element: Element to read

    Returns:
        Collapsed text content
    """
    return WHITESPACE.sub(" ", element.text_content()).strip()

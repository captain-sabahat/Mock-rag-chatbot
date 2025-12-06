"""
===============================================================================
parsers/markdown_parser.py
===============================================================================

SUMMARY:
- Implements Markdown parsing, converts to plain text
- Inherits from BaseParser
- Uses markdown library for rendering/stripping
- Placeholder for performance and resource metrics
- Designed for easy extension, e.g., extracting tables

WORK:
- Inputs:
  - content (bytes, Markdown)
- Outputs:
  - cleaned_text (str)
- Dependencies:
  - markdown library
  - config control for options
- Future:
  - Extract metadata, links, images
  - Convert to HTML or plain text

MLFLOW:
- parse_time_ms
- resource metrics
- success flag

SENSITIVE LINES:
- Markdown parsing
- HTML conversions
- Exception safety in parsing

"""

import markdown
from bs4 import BeautifulSoup
from .base_parser import BaseParser
import time

class MarkdownParser(BaseParser):
    """
    Converts markdown to clean text, stripping HTML tags.
    """

    @property
    def name(self) -> str:
        return "markdown_parser"

    async def parse(self, content: bytes, **kwargs) -> str:
        """
        Parse markdown bytes into clean text.
        """
        start_time = time.perf_counter()
        # #MLFLOW:parse_time_ms -- start

        try:
            md_text = content.decode(self.config.get("encoding", "utf-8"))
            html = markdown.markdown(md_text)
            # Remove HTML tags to get clean text
            soup = BeautifulSoup(html, "html.parser")
            cleaned_text = soup.get_text(separator=" ", strip=True)
            # #MLFLOW:success_flag -- True
            return cleaned_text
        except Exception as e:
            # #MLFLOW:parse_success_flag -- False
            raise e
        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000
            # #MLFLOW:parse_time_ms -- record duration

# Future:
# - Add options for HTML output
# - Embed image/audio extraction
# - Track resource usage for MLflow


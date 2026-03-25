"""Extract plain text from PDF, DOCX, or TXT uploads."""
import io
from typing import Tuple, Set
import chardet


class UnsupportedFormatError(Exception):
    """Raised when document format is not supported."""
    pass


class CorruptedFileError(Exception):
    """Raised when document file is corrupted or cannot be parsed."""
    pass


class DocumentParser:
    """Handles text extraction from multiple document formats with error handling."""
    
    SUPPORTED_FORMATS: Set[str] = {"pdf", "docx", "txt"}
    CHARS_PER_PAGE: int = 3000  # Estimate for page count calculation
    
    def __init__(self):
        """Initialize the DocumentParser."""
        pass
    
    async def extract_text(self, filename: str, content: bytes) -> Tuple[str, int]:
        """
        Extract text from document and estimate page count.
        
        Args:
            filename: Name of the uploaded file
            content: Raw bytes of the file content
            
        Returns:
            Tuple of (extracted_text, estimated_page_count)
            
        Raises:
            UnsupportedFormatError: If file format is not supported
            CorruptedFileError: If file is corrupted or cannot be parsed
        """
        if not filename or not content:
            raise ValueError("Filename and content are required")
        
        # Extract file extension
        ext = self._get_file_extension(filename)
        
        # Validate format
        if ext not in self.SUPPORTED_FORMATS:
            raise UnsupportedFormatError(
                f"Unsupported format '.{ext}'. Supported formats: {', '.join(sorted(self.SUPPORTED_FORMATS))}"
            )
        
        # Route to appropriate parser
        try:
            if ext == "pdf":
                return self._parse_pdf(content)
            elif ext == "docx":
                return self._parse_docx(content)
            elif ext == "txt":
                return self._parse_txt(content)
            else:
                raise UnsupportedFormatError(f"Format '.{ext}' not implemented")
        except (UnsupportedFormatError, CorruptedFileError):
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            # Wrap unexpected errors as corrupted file errors
            raise CorruptedFileError(f"Failed to parse {ext.upper()} file: {str(e)}")
    
    def _get_file_extension(self, filename: str) -> str:
        """Extract and normalize file extension."""
        parts = filename.lower().rsplit(".", 1)
        if len(parts) < 2:
            raise UnsupportedFormatError("File has no extension")
        return parts[-1]
    
    def _parse_pdf(self, content: bytes) -> Tuple[str, int]:
        """
        Parse PDF document and extract text.
        
        Args:
            content: Raw PDF file bytes
            
        Returns:
            Tuple of (extracted_text, page_count)
            
        Raises:
            CorruptedFileError: If PDF is corrupted or cannot be read
        """
        try:
            import PyPDF2
        except ImportError:
            raise CorruptedFileError("PyPDF2 library not available")
        
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(content))
            
            # Check if PDF is encrypted
            if reader.is_encrypted:
                raise CorruptedFileError("PDF is encrypted and cannot be processed")
            
            pages = len(reader.pages)
            if pages == 0:
                raise CorruptedFileError("PDF contains no pages")
            
            # Extract text from all pages
            text_parts = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            
            text = "\n\n".join(text_parts)
            
            # Validate extracted text
            if not text.strip():
                raise CorruptedFileError("PDF contains no extractable text")
            
            return text, pages
            
        except PyPDF2.errors.PdfReadError as e:
            raise CorruptedFileError(f"PDF read error: {str(e)}")
        except Exception as e:
            if isinstance(e, CorruptedFileError):
                raise
            raise CorruptedFileError(f"PDF parsing failed: {str(e)}")
    
    def _parse_docx(self, content: bytes) -> Tuple[str, int]:
        """
        Parse DOCX document and extract text.
        
        Args:
            content: Raw DOCX file bytes
            
        Returns:
            Tuple of (extracted_text, estimated_page_count)
            
        Raises:
            CorruptedFileError: If DOCX is corrupted or cannot be read
        """
        try:
            import docx
        except ImportError:
            raise CorruptedFileError("python-docx library not available")
        
        try:
            doc = docx.Document(io.BytesIO(content))
            
            # Extract text from paragraphs
            text_parts = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_parts.append(paragraph.text)
            
            text = "\n\n".join(text_parts)
            
            # Validate extracted text
            if not text.strip():
                raise CorruptedFileError("DOCX contains no extractable text")
            
            # Estimate page count
            pages = max(1, len(text) // self.CHARS_PER_PAGE)
            
            return text, pages
            
        except Exception as e:
            if isinstance(e, CorruptedFileError):
                raise
            raise CorruptedFileError(f"DOCX parsing failed: {str(e)}")
    
    def _parse_txt(self, content: bytes) -> Tuple[str, int]:
        """
        Parse plain text document.
        
        Args:
            content: Raw text file bytes
            
        Returns:
            Tuple of (extracted_text, estimated_page_count)
            
        Raises:
            CorruptedFileError: If text cannot be decoded
        """
        try:
            # Try UTF-8 first
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError:
                # Fall back to chardet for encoding detection
                detected = chardet.detect(content)
                encoding = detected.get("encoding", "utf-8")
                text = content.decode(encoding, errors="replace")
            
            # Validate extracted text
            if not text.strip():
                raise CorruptedFileError("Text file is empty")
            
            # Estimate page count
            pages = max(1, len(text) // self.CHARS_PER_PAGE)
            
            return text, pages
            
        except Exception as e:
            if isinstance(e, CorruptedFileError):
                raise
            raise CorruptedFileError(f"Text file parsing failed: {str(e)}")


# Maintain backward compatibility with existing code
async def extract_text(filename: str, content: bytes) -> Tuple[str, int]:
    """
    Legacy function for backward compatibility.
    Returns (full_text, estimated_page_count).
    """
    parser = DocumentParser()
    return await parser.extract_text(filename, content)

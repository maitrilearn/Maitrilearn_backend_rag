"""
Input validation utilities.
All public-facing endpoints should validate inputs before processing.
"""


class ValidationError(Exception):
    """Raised when input validation fails."""
    def __init__(self, message: str, field: str = None):
        self.message = message
        self.field   = field
        super().__init__(message)


def validate_text(text, field="text", min_len=1, max_len=2000):
    """Validate a text input field."""
    if not text or not isinstance(text, str):
        raise ValidationError(f"{field} is required", field)

    text = text.strip()

    if len(text) < min_len:
        raise ValidationError(f"{field} is too short (min {min_len} chars)", field)

    if len(text) > max_len:
        raise ValidationError(
            f"{field} is too long (max {max_len} chars, got {len(text)})", field
        )

    return text


def validate_topic(topic):
    """Validate a topic/subject field."""
    return validate_text(topic, field="topic", min_len=2, max_len=200)


def validate_question(question):
    """Validate a student question."""
    return validate_text(question, field="question", min_len=3, max_len=1000)


def validate_command(command):
    """Validate a terminal command."""
    return validate_text(command, field="command", min_len=1, max_len=500)


def validate_feedback(feedback):
    """Validate feedback text."""
    return validate_text(feedback, field="feedback", min_len=5, max_len=2000)


def validate_filename(filename):
    """Validate a Supabase storage filename."""
    if not filename or not isinstance(filename, str):
        raise ValidationError("filename is required", "filename")

    filename = filename.strip()

    # Block path traversal
    if ".." in filename or "/" in filename or "\\" in filename:
        raise ValidationError("Invalid filename — path traversal not allowed", "filename")

    # Must have extension
    if "." not in filename:
        raise ValidationError("filename must have an extension (e.g. .pdf, .txt)", "filename")

    allowed = {".pdf", ".txt", ".md"}
    ext = "." + filename.rsplit(".", 1)[-1].lower()
    if ext not in allowed:
        raise ValidationError(
            f"File type not supported. Allowed: {', '.join(allowed)}", "filename"
        )

    return filename

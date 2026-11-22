
import json
import random
import re
import threading
import time
import uuid
from datetime import datetime, timezone

from google.genai import types


# ============================================================
# 1. LLM Configuration
# ============================================================

GEMINI_MODEL = "gemini-3.5-flash-lite"
TEMPERATURE = 0.0
DEFAULT_MAX_OUTPUT_TOKENS = 2048

# API retry configuration
MAX_API_RETRIES = 4
RETRY_BASE_DELAY = 2
RETRY_MAX_DELAY = 30

# Minimum interval between API calls
MIN_API_INTERVAL = 5.0

API_CALL_LOCK = threading.Lock()
LAST_API_CALL_TIME = 0.0


# ============================================================
# 2. API Rate Limiting
# ============================================================

def wait_for_api_slot():
    """
    Ensure a minimum interval between API calls.

    The lock is held while waiting to prevent concurrent calls
    from violating the minimum interval.
    """
    global LAST_API_CALL_TIME

    with API_CALL_LOCK:
        elapsed = time.time() - LAST_API_CALL_TIME
        wait_time = MIN_API_INTERVAL - elapsed

        if wait_time > 0:
            time.sleep(wait_time)

        LAST_API_CALL_TIME = time.time()


# ============================================================
# 3. Retryable Error Detection
# ============================================================

def is_retryable_error(error):
    """
    Determine whether an API error should be retried.

    Retry conditions are identified using error-message keywords.
    """
    error_message = str(error).lower()

    retryable_indicators = [
        "429",
        "rate limit",
        "resource exhausted",
        "quota",
        "503",
        "service unavailable",
        "500",
        "internal server error",
        "timeout",
        "temporarily unavailable",
    ]

    return any(
        indicator in error_message
        for indicator in retryable_indicators
    )


# ============================================================
# 4. LLM Call Logging
# ============================================================

llm_call_logs = []


def utc_now():
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def build_llm_log_record(
    call_id,
    execution_id,
    policy,
    prompt,
    max_output_tokens,
    started_at,
    completed_at,
    response_text=None,
    error=None,
    attempts=None,
    usage_metadata=None,
    task=None,
    question_id=None,
):
    """Build a structured record for a single logical LLM call."""

    return {
        "call_id": call_id,
        "execution_id": execution_id,
        "policy": policy,
        "task": task,
        "question_id": question_id,
        "prompt": prompt,
        "response": response_text,
        "model": GEMINI_MODEL,
        "temperature": TEMPERATURE,
        "max_output_tokens": max_output_tokens,
        "started_at": started_at,
        "completed_at": completed_at,
        "timestamp": completed_at,
        "status": "failed" if error else "success",
        "error": str(error) if error else None,
        "attempts": attempts or [],
        "usage_metadata": usage_metadata,
    }


def save_llm_log(record):
    """
    Save the LLM call record.

    The log is retained in memory even if persistent logging fails.
    """
    try:
        from src.experiment_logger import append_jsonl

        append_jsonl(record, "llm_calls.jsonl")

    finally:
        llm_call_logs.append(record)


# ============================================================
# 5. Shared LLM Generation Function
# ============================================================

def generate_llm(
    prompt,
    execution_id=None,
    policy=None,
    task=None,
    question_id=None,
    max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
):
    """
    Generate a response using Gemini.

    Returns:
        tuple[str, str]:
            response_text, call_id
    """

    from src.llm_client import client

    # Use the current experiment RUN_ID when no execution_id is provided.
    if execution_id is None:
        from src.experiment_logger import RUN_ID
        execution_id = RUN_ID

    call_id = str(uuid.uuid4())
    started_at = utc_now()

    attempts = []
    response = None
    last_error = None

    # --------------------------------------------------------
    # API execution with retry
    # --------------------------------------------------------

    for attempt_num in range(MAX_API_RETRIES + 1):

        wait_for_api_slot()

        attempt_started_at = utc_now()

        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=TEMPERATURE,
                    max_output_tokens=max_output_tokens,
                ),
            )

            response_text = response.text

            if not response_text:
                raise ValueError("Empty response from Gemini API.")

            attempts.append({
                "attempt": attempt_num + 1,
                "status": "success",
                "started_at": attempt_started_at,
                "completed_at": utc_now(),
            })

            last_error = None
            break

        except Exception as error:

            attempts.append({
                "attempt": attempt_num + 1,
                "status": "failed",
                "started_at": attempt_started_at,
                "completed_at": utc_now(),
                "error": str(error),
            })

            last_error = error

            # Do not retry non-retryable errors.
            if not is_retryable_error(error):
                break

            # Stop if retry budget is exhausted.
            if attempt_num >= MAX_API_RETRIES:
                break

            # Exponential backoff with jitter.
            delay = min(
                RETRY_BASE_DELAY * (2 ** attempt_num),
                RETRY_MAX_DELAY,
            )

            delay += random.uniform(0, 1)

            print(
                f"[Retry {attempt_num + 1}/{MAX_API_RETRIES}] "
                f"API error: {error}"
            )

            time.sleep(delay)

    # --------------------------------------------------------
    # Failure logging
    # --------------------------------------------------------

    if last_error is not None:

        record = build_llm_log_record(
            call_id=call_id,
            execution_id=execution_id,
            policy=policy,
            prompt=prompt,
            max_output_tokens=max_output_tokens,
            started_at=started_at,
            completed_at=utc_now(),
            error=last_error,
            attempts=attempts,
            task=task,
            question_id=question_id,
        )

        try:
            save_llm_log(record)

        except Exception as log_error:
            print(f"[Warning] Failed to save LLM log: {log_error}")

        raise last_error

    # --------------------------------------------------------
    # Successful response logging
    # --------------------------------------------------------

    usage_metadata = None

    if response is not None:
        usage = getattr(response, "usage_metadata", None)

        if usage is not None:
            usage_metadata = {
                "prompt_token_count": getattr(
                    usage, "prompt_token_count", None
                ),
                "candidates_token_count": getattr(
                    usage, "candidates_token_count", None
                ),
                "total_token_count": getattr(
                    usage, "total_token_count", None
                ),
            }

    record = build_llm_log_record(
        call_id=call_id,
        execution_id=execution_id,
        policy=policy,
        prompt=prompt,
        max_output_tokens=max_output_tokens,
        started_at=started_at,
        completed_at=utc_now(),
        response_text=response_text,
        attempts=attempts,
        usage_metadata=usage_metadata,
        task=task,
        question_id=question_id,
    )

    try:
        save_llm_log(record)

    except Exception as log_error:
        # Logging failure must not trigger another API generation.
        print(f"[Warning] Failed to save LLM log: {log_error}")

    return response_text, call_id


# ============================================================
# 6. Answer Extraction Prompt
# ============================================================

ANSWER_EXTRACTION_PROMPT = """
Extract the core answer from the following text.

Rules:
- Return only the answer itself.
- Do not include explanations.
- Do not include quotation marks.
- Preserve names, dates, and important details.
- If the text contains multiple possible answers, choose the
  answer that most directly addresses the original question.

Original question:
{question}

Generated answer:
{answer}
"""


# ============================================================
# 7. Answer Cleaning
# ============================================================

def clean_extracted_answer(raw):
    """
    Clean an extracted answer for evaluation.

    Removes:
    - Markdown code fences
    - Leading/trailing whitespace
    - Common answer prefixes
    - Surrounding quotation marks
    """

    if raw is None:
        return ""

    text = str(raw).strip()

    # Remove Markdown code fences.
    text = re.sub(
        r"```(?:\w+)?",
        "",
        text,
    )

    # Remove common answer prefixes.
    text = re.sub(
        r"^(?:answer|the answer is|final answer)\s*[:\-]\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Remove surrounding quotation marks.
    text = text.strip("\"'`")

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()

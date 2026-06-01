from __future__ import annotations

import os
from typing import Optional

from tenacity import retry, stop_after_attempt, wait_random_exponential


MMBENCH_PROMPT = """You are an AI assistant who will help me to evaluate the response given the question and the correct answer.
To mark a response, you should output a single integer between 1 and 5 (including 1, 5).
5 means that the response perfectly matches the answer.
1 means that the response is completely different from the answer.

Example 1:
Question: Is it overcast?
Answer: no
Response: yes
Your mark: 1

Example 2:
Question: Who is standing at the table?
Answer: woman
Response: Jessica
Your mark: 3

Example 3:
Question: Are there drapes to the right of the bed?
Answer: yes
Response: yes
Your mark: 5

Your Turn:
Question: {question}
Answer: {answer}
Response: {prediction}"""


MMBENCH_EXTRA_PROMPT = """You are an AI assistant who will help me to evaluate the response given the question, the correct answer, and extra answers that are also correct.
To mark a response, you should output a single integer between 1 and 5 (including 1, 5).
5 means that the response perfectly matches the answer or any of the extra answers.
1 means that the response is completely different from the answer and all of the extra answers.

Example 1:
Question: Is it overcast?
Answer: no
Extra Answers: ['doesn't look like it', 'no',' it's sunny']
Response: yes
Your mark: 1

Example 2:
Question: Who is standing at the table?
Answer: woman
Extra Answers: ['a woman', 'a lady', 'woman']
Response: Jessica
Your mark: 3

Example 3:
Question: Are there drapes to the right of the bed?
Answer: yes
Extra Answers: ['yes, there are drapes', 'yeah', 'the drapes are to the right of the king bed']
Response: yes
Your mark: 5

Your Turn:
Question: {question}
Answer: {answer}
Extra Answers: {extra_answers}
Response: {prediction}"""


def parse_score(output: str, tag: str = "Your mark:") -> int:
    output = str(output).strip()
    if output.isdigit():
        return _validate_score(int(output), output)

    start_idx = output.find(tag)
    if start_idx == -1:
        raise ValueError(f"Invalid LLM-Match output string: {output}")

    end_idx = output.find("\n", start_idx)
    if end_idx == -1:
        score_text = output[start_idx:].replace(tag, "").strip()
    else:
        score_text = output[start_idx:end_idx].replace(tag, "").strip()
    return _validate_score(int(score_text), output)


def get_llm_match_score(
    question: str,
    answer: str,
    prediction: str,
    extra_answers: Optional[list] = None,
    openai_key: Optional[str] = None,
    openai_model: str = "gpt-4-1106-preview",
    openai_seed: int = 1234,
    openai_max_tokens: int = 32,
    openai_temperature: float = 0.2,
    verbose: bool = False,
) -> int:
    """Score one answer with OpenEQA-compatible LLM-Match.

    This intentionally mirrors the upstream OpenEQA LLM-Match behavior so the
    spatial memory harness can run independently from the upstream package.
    """

    if prediction is None:
        return 0

    prompt = MMBENCH_PROMPT if extra_answers is None else MMBENCH_EXTRA_PROMPT
    content = prompt.format(
        question=question,
        answer=answer,
        prediction=prediction,
        extra_answers=extra_answers,
    )
    output = call_openai_chat_completion(
        messages=[{"role": "user", "content": content}],
        api_key=openai_key,
        model=openai_model,
        seed=openai_seed,
        max_tokens=openai_max_tokens,
        temperature=openai_temperature,
        verbose=verbose,
    )
    return parse_score(output)


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def call_openai_chat_completion(
    *,
    messages: list[dict[str, str]],
    api_key: Optional[str],
    model: str,
    seed: Optional[int],
    max_tokens: int,
    temperature: float,
    verbose: bool,
) -> str:
    from openai import OpenAI

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is required for LLM-Match scoring. "
            "Set the environment variable or pass --openai-key."
        )

    client = OpenAI(api_key=key)
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        seed=seed,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if verbose:
        print(f"openai api response: {completion}")
    if len(completion.choices) != 1:
        raise RuntimeError(f"Expected one OpenAI choice, got {len(completion.choices)}")
    return completion.choices[0].message.content or ""


def _validate_score(score: int, raw_output: str) -> int:
    if score < 1 or score > 5:
        raise ValueError(f"LLM-Match score must be in [1, 5], got {score}: {raw_output}")
    return score

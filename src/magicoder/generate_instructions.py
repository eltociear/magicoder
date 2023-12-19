import asyncio
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from datasets import Dataset, load_dataset
from tqdm.auto import tqdm
from transformers import HfArgumentParser

import magicoder

# DO NOT CHANGE THE FOLLOWING
ERROR_MARGIN = 10
NUM_WORDS_CHOICES = ["0--200", "200--400", "400--600", "600--800"]


@dataclass(frozen=True)
class Args:
    seed_data_files: list[str] = field(
        metadata={"help": "Path to the seed code snippets"}
    )
    seed_code_start_index: int
    # `seed_code_start_index` + `max_new_data` is the last-to-end seed code index
    max_new_data: int
    continue_from: str | None = field(default=None)

    # Keep the following arguments unchanged for reproducibility
    seed: int = field(default=976)

    temperature: float = field(default=0.7)
    model: str = field(default="gpt-3.5-turbo-1106")
    model_max_tokens: int = field(default=8192)
    max_new_tokens: int = field(default=2500)

    rpm: int = field(
        default=1, metadata={"help": "Requests per minute to the model API"}
    )
    delay: int | None = field(
        default=None, metadata={"help": "Delay between batched requests in seconds"}
    )

    tag: str = field(
        default="",
        metadata={
            "help": "Custom tag as part of the output filename, not affecting the fingerprint"
        },
    )

    def fingerprint(self, sys_prompt: str, user_prompt: str) -> str:
        # The combination of arguments can uniquely determine the generation process
        args = (
            self.seed_data_files,
            self.seed,
            self.temperature,
            self.model,
            self.model_max_tokens,
            sys_prompt,
            user_prompt,
            ERROR_MARGIN,
            NUM_WORDS_CHOICES,
        )
        return magicoder.utils.compute_fingerprint(*args, hash_length=5)


def parse_instruction(response_text: str) -> str | None:
    count = response_text.count('"""')
    if count == 0:
        response_text = response_text.strip()
    elif count == 2:
        start_index = response_text.index('"""')
        end_index = response_text.rindex('"""')
        response_text = response_text[start_index + 3 : end_index].strip()
    else:
        return None
    if response_text.startswith("```") and response_text.endswith("```"):
        response_text = response_text[3:-3].strip()
    return response_text


async def main():
    args = cast(Args, HfArgumentParser(Args).parse_args_into_dataclasses()[0])
    openai_client = magicoder.utils.OpenAIClient()
    raw_dataset: Dataset = load_dataset(
        "json",
        data_files=args.seed_data_files,
        split="train",
        num_proc=magicoder.utils.N_CORES,
    )

    # Every run should produce the same data as long as the default params are not changed
    start_index = args.seed_code_start_index
    end_index = min(start_index + args.max_new_data, len(raw_dataset))
    raw_dataset = raw_dataset.select(range(start_index, end_index))
    dataset = raw_dataset.to_list()

    sys_prompt_template = Path("data/prompt_system.txt").read_text()
    user_prompt_template = Path("data/prompt_user.txt").read_text()
    timestamp = magicoder.utils.timestamp()
    data_fingerprint = args.fingerprint(sys_prompt_template, user_prompt_template)
    if args.continue_from is not None:
        assert (
            data_fingerprint in args.continue_from
        ), f"Fingerprint mismatch: {data_fingerprint}"
        assert f"{start_index}_{end_index}" in args.continue_from, "Index mismatch"
        old_path = Path(args.continue_from)
        assert old_path.exists()
        old_data = magicoder.utils.read_jsonl(old_path)
        assert len(old_data) > 0
        last_seed = old_data[-1]["seed"]
        # Find seed
        seed_index = next(
            idx for idx, d in enumerate(dataset) if d["seed"] == last_seed
        )
        n_skipped = seed_index - start_index + 1
        # n_skipped = last_index - start_index + 1
        print("Continuing from", old_path)
        f_out = old_path.open("a")
    else:
        tag = "" if args.tag == "" else f"-{args.tag}"
        path = Path(
            f"data{tag}-{data_fingerprint}-{start_index}_{end_index}-{timestamp}.jsonl"
        )
        assert not path.exists()
        f_out = path.open("w")
        print("Saving to", path)
        n_skipped = 0
    dataset = dataset[n_skipped:]
    chunked_dataset = list(magicoder.utils.chunked(dataset, n=args.rpm))
    pbar = tqdm(chunked_dataset)
    n_succeeded = 0
    for chunk_index, examples in enumerate(pbar):
        # map to the index in the original seed snippets
        effective_index = chunk_index * args.rpm + start_index + n_skipped
        if chunk_index > 0 and args.rpm != 1:
            print("Sleeping for 60 seconds...")
            time.sleep(60)
        # assert index + start_index == example["index"]
        request_params = list[dict[str, Any]]()
        attr_num_words: list[str] = []
        for index, example in enumerate(examples):
            seed = args.seed + effective_index + index
            random.seed(seed)
            num_words = random.choice(NUM_WORDS_CHOICES)
            attr_num_words.append(num_words)
            sys_prompt = sys_prompt_template
            user_prompt = user_prompt_template.format(
                code=example["seed"], num_words=num_words
            )
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ]
            # Make sure the generation is within the context size of the model
            max_new_tokens = min(
                args.max_new_tokens,
                args.model_max_tokens
                - magicoder.utils.num_tokens_from_string(
                    f"{sys_prompt}\n{user_prompt}", args.model
                )
                # error margin (e.g., due to conversation tokens)
                - ERROR_MARGIN,
            )
            # if max_new_tokens <= 0:
            #     continue
            params = dict(
                model=args.model,
                messages=messages,
                max_tokens=max_new_tokens,
                n=1,
                temperature=args.temperature,
                seed=seed,
            )
            request_params.append(params)
        assert len(request_params) == len(examples)
        print(f"Ready to make {len(request_params)} requests")
        responses = await openai_client.dispatch_chat_completions(
            request_params, delay=args.delay
        )
        assert len(examples) == len(responses)
        for num_words, example, response in zip(attr_num_words, examples, responses):
            if isinstance(response, BaseException):
                print("Exception when generating response:", response)
                continue
            choice = response.choices[0]
            if choice.finish_reason != "stop":
                print("Failed to generate a complete response")
                continue
            instruction = parse_instruction(choice.message.content)
            if instruction is None:
                continue
            fingerprint = response.system_fingerprint
            assert fingerprint is not None
            data = dict(
                num_words=num_words,
                instruction=instruction,
                openai_fingerprint=fingerprint,
                **example,
            )

            print(
                "[Seed]",
                example["seed"],
                "[Num Words]",
                num_words,
                "[Instruction]",
                instruction,
                sep="\n",
                end="\n\n",
            )
            n_succeeded += 1
            f_out.write(json.dumps(data) + "\n")
            f_out.flush()
        total_requests = chunk_index * args.rpm + len(examples)
        pbar.set_description(f"Success ratio: {n_succeeded} / {total_requests}")


if __name__ == "__main__":
    asyncio.run(main())

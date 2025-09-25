# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from .processor_utils import DatasetProcessor, greedy_knapsack, infer_seqlen
import torch


if TYPE_CHECKING:
    from ..mm_plugin import AudioInput, ImageInput, VideoInput


logger = logging.get_logger(__name__)


@dataclass
class SupervisedDatasetProcessor(DatasetProcessor):

    def _encode_data_example(
        self,
        prompt: List[Dict[str, str]],
        response: List[Dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        # 🔧 Stage2 修正: 单独处理简单的 "图->文" 任务
        # 判断条件：单轮对话，且为指定的Stage2数据集
        is_stage2_pretrain = (
            len(prompt) == 1
            and len(response) == 1
            and self.data_args.dataset is not None
            and any("pathtext_stage2" in d for d in self.data_args.dataset)
        )

        if is_stage2_pretrain:
            # For Stage 2, expand the single <image> placeholder to a sequence of placeholders.
            # We add 1 to the length to account for the [CLS] token that is prepended by PruMerge.
            image_seq_len = getattr(self.data_args, "image_seq_length", 1536) + 1
            if image_seq_len > 1:
                image_placeholder = getattr(self.template, "image_placeholder", "<image>")
                # Assuming the placeholder is in the first turn of the prompt
                original_content = prompt[0]["content"]
                if image_placeholder in original_content:
                    expanded_content = original_content.replace(image_placeholder, image_placeholder * image_seq_len)
                    prompt[0]["content"] = expanded_content
                
                # Also expand the images list to match the number of placeholders
                if images and len(images) == 1:
                    images = images * image_seq_len

            # 彻底绕开 encode_multiturn，使用更底层的 _encode 来确保输入/标签分离
            
            # 1. 单独编码 prompt (USER: <image>...<image>\nquestion)
            #    注意：这里的 message 只包含 prompt，不包含 response
            prompt_messages = self.template.mm_plugin.process_messages(prompt, images, videos, audios, self.processor)
            encoded_prompt = self.template._encode(self.tokenizer, prompt_messages, system, tools)
            prompt_ids = []
            for ids in encoded_prompt:
                prompt_ids.extend(ids)

            # 2. 单独编码 response (ASSISTANT: caption)
            #    注意：这里的 message 只包含 response
            #    为了让模板正确添加 "ASSISTANT:" 等前缀，我们需要一个假的空 prompt
            dummy_prompt = [{"role": "user", "content": ""}]
            response_messages = self.template.mm_plugin.process_messages(dummy_prompt + response, [], [], [], self.processor)
            encoded_response = self.template._encode(self.tokenizer, response_messages, system, tools)
            
            # encoded_response 此时是 [[-100,...], [assistant_prefix, caption_ids]]
            # 我们只需要第二部分
            response_ids = encoded_response[1]
            
            # 3. 最终拼接
            input_ids = prompt_ids + response_ids
            labels = [IGNORE_INDEX] * len(prompt_ids) + response_ids

            # 截断处理
            if len(input_ids) > self.data_args.cutoff_len:
                input_ids = input_ids[:self.data_args.cutoff_len]
                labels = labels[:self.data_args.cutoff_len]
        
            attention_mask = None
            if self.data_args.use_prefix_lm_mask:
                # Prefix-LM attention mask (float tensor, 0.0 for visible, large negative for masked)
                prompt_len = len(prompt_ids)
                if prompt_len > self.data_args.cutoff_len:
                    prompt_len = self.data_args.cutoff_len

                seq_len = len(input_ids)
                large_negative_value = -1e9
                attention_mask = torch.full((seq_len, seq_len), large_negative_value, dtype=torch.float32)
                causal_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool))
                attention_mask.masked_fill_(causal_mask, 0.0)
                attention_mask[:, :prompt_len] = 0.0 # Ensure all tokens can see the full prompt
            
            return input_ids, labels, attention_mask

        else:
            # --- Stage 3: General logic for VQA and other tasks ---

            # --- FIX: Replicate image token expansion for Stage 3 ---
            # This logic is copied from Stage 2 to align the data pipeline.
            image_seq_len = getattr(self.data_args, "image_seq_length", 0)
            if image_seq_len >= 0:
                effective_len = image_seq_len + 1  # Adhere to the convention: num_patches + 1 for [CLS]
                image_placeholder = getattr(self.template, "image_placeholder", "<image>")
                if prompt and image_placeholder in prompt[0]["content"]:
                    original_content = prompt[0]["content"]
                    expanded_content = original_content.replace(image_placeholder, image_placeholder * effective_len)
                    prompt[0]["content"] = expanded_content
                
                if images and len(images) == 1:
                    images = images * effective_len
            # --- END OF FIX ---

            # Now, the standard Stage 3 logic will process the prompt with expanded image tokens.
            # --- START: Simplified logic mirrored from Stage 2 for consistent experiment ---
            prompt_messages = self.template.mm_plugin.process_messages(prompt, images, videos, audios, self.processor)
            encoded_prompt = self.template._encode(self.tokenizer, prompt_messages, system, tools)
            prompt_ids = []
            for ids in encoded_prompt:
                prompt_ids.extend(ids)

            dummy_prompt = [{"role": "user", "content": ""}]
            response_messages = self.template.mm_plugin.process_messages(dummy_prompt + response, [], [], [], self.processor)
            encoded_response = self.template._encode(self.tokenizer, response_messages, system, tools)
            
            response_ids = encoded_response[1]
            
            input_ids = prompt_ids + response_ids
            labels = [IGNORE_INDEX] * len(prompt_ids) + response_ids

            if len(input_ids) > self.data_args.cutoff_len:
                input_ids = input_ids[:self.data_args.cutoff_len]
                labels = labels[:self.data_args.cutoff_len]
                
            # 4. 为 Stage 3 生成 Attention Mask
            attention_mask = None
            if self.data_args.use_prefix_lm_mask:
                prompt_len_stage3 = 0
                for label in labels:
                    if label == IGNORE_INDEX:
                        prompt_len_stage3 += 1
                    else:
                        break
                
                seq_len = len(input_ids)
                if prompt_len_stage3 >= seq_len:
                    prompt_len_stage3 = seq_len - 1

                large_negative_value = -1e9
                attention_mask = torch.full((seq_len, seq_len), large_negative_value, dtype=torch.float32)
                causal_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool))
                attention_mask.masked_fill_(causal_mask, 0.0)
                attention_mask[:, :prompt_len_stage3] = 0.0

            return input_ids, labels, attention_mask

    def _process_single_example(
        self,
        example: Dict[str, Any],
    ) -> Dict[str, List[int]]:
        processed_example = self._encode_data_example(
            prompt=example["_prompt"],
            response=example["_response"],
        )
        if not processed_example["input_ids"]:
            return {"input_ids": []}

        if self.data_args.mm_flatten_image_to_one:
            example["_image"] = [img for img in example["_image"] if img is not None]
            if len(example["_image"]) == 0:
                return {"input_ids": []}

            processed_example["image"] = example["_image"][0]

        return processed_example

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
        # for multiturn examples, we only mask the prompt part in each prompt-response pair.
        model_inputs = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            input_ids, labels, attention_mask = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
            )
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append(attention_mask)
            model_inputs["labels"].append(labels)
            model_inputs["images"].append(examples["_images"][i])
            model_inputs["videos"].append(examples["_videos"][i])
            model_inputs["audios"].append(examples["_audios"][i])

        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:\n{}".format(example["labels"]))
        print(f"labels:\n{self.tokenizer.decode(valid_labels, skip_special_tokens=False)}")


@dataclass
class PackedSupervisedDatasetProcessor(SupervisedDatasetProcessor):
    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # TODO: use `position_ids` to achieve packing
        # build inputs with format `<bos> X1 Y1 <eos> <bos> X2 Y2 <eos>`
        # and labels with format `<ignore> ... <ignore> Y1 <eos> <ignore> ... <ignore> Y2 <eos>`
        valid_num = 0
        batch_input_ids, batch_labels, batch_images, batch_videos, batch_audios = [], [], [], [], []
        lengths = []
        length2indexes = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            input_ids, labels, attention_mask = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
            )
            length = len(input_ids)
            if length > self.data_args.cutoff_len:
                logger.warning_rank0(f"Dropped lengthy example with length {length} > {self.data_args.cutoff_len}.")
            else:
                lengths.append(length)
                length2indexes[length].append(valid_num)
                batch_input_ids.append(input_ids)
                batch_labels.append(labels)
                batch_images.append(examples["_images"][i] or [])
                batch_videos.append(examples["_videos"][i] or [])
                batch_audios.append(examples["_audios"][i] or [])
                valid_num += 1

        model_inputs = defaultdict(list)
        knapsacks = greedy_knapsack(lengths, self.data_args.cutoff_len)
        for knapsack in knapsacks:
            packed_input_ids, packed_attention_masks, packed_position_ids, packed_labels = [], [], [], []
            packed_images, packed_videos, packed_audios = [], [], [], []
            for i, length in enumerate(knapsack):
                index = length2indexes[length].pop()
                packed_input_ids += batch_input_ids[index]
                packed_position_ids += list(range(len(batch_input_ids[index])))  # NOTE: pad_to_multiple_of ignore this
                packed_labels += batch_labels[index]
                packed_images += batch_images[index]
                packed_videos += batch_videos[index]
                packed_audios += batch_audios[index]
                if self.data_args.neat_packing:
                    packed_attention_masks += [i + 1] * len(batch_input_ids[index])  # start from 1
                else:
                    packed_attention_masks += [1] * len(batch_input_ids[index])

            if len(packed_input_ids) < self.data_args.cutoff_len + 1:  # avoid flash_attn drops attn mask
                pad_length = self.data_args.cutoff_len - len(packed_input_ids) + 1
                packed_input_ids += [self.tokenizer.pad_token_id] * pad_length
                packed_position_ids += [0] * pad_length
                packed_labels += [IGNORE_INDEX] * pad_length
                if self.data_args.neat_packing:
                    packed_attention_masks += [0] * pad_length
                else:
                    packed_attention_masks += [1] * pad_length  # more efficient flash_attn

            if len(packed_input_ids) != self.data_args.cutoff_len + 1:
                raise ValueError("The length of packed example should be identical to the cutoff length.")

            model_inputs["input_ids"].append(packed_input_ids)
            model_inputs["attention_mask"].append(packed_attention_masks)
            model_inputs["position_ids"].append(packed_position_ids)
            model_inputs["labels"].append(packed_labels)
            model_inputs["images"].append(packed_images or None)
            model_inputs["videos"].append(packed_videos or None)
            model_inputs["audios"].append(packed_audios or None)

        return model_inputs

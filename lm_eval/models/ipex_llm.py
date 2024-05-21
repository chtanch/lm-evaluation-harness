import copy
import os
from datetime import timedelta
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

import ipex_llm.transformers

import torch
# import torch.nn.functional as F
# import transformers
# from accelerate import (
#     Accelerator,
#     DistributedType,
#     InitProcessGroupKwargs,
#     find_executable_batch_size,
# )
# from huggingface_hub import HfApi
# from packaging import version
# from peft import PeftModel
# from peft import __version__ as PEFT_VERSION
# from tqdm import tqdm
# from transformers.models.auto.modeling_auto import (
#     MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
#     MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
# )

from lm_eval import utils
from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import (
    Collator,
    clear_torch_cache,
    get_dtype,
    pad_and_concat,
    stop_sequences_criteria,
)
from lm_eval.models.huggingface import HFLM

eval_logger = utils.eval_logger

@register_model("ipex-llm")
class IPEXLLM(HFLM):
    """
    An abstracted Huggingface model class. Enables usage with both models of
    `transformers.AutoModelForCausalLM` and `transformers.AutoModelForSeq2SeqLM` classes.

    Currently does not support data-parallel multi-GPU with HF Accelerate.
    """

    def __init__(self, *args, **kwargs) -> None:
        # HFLM does not detect xpu and sets self.device to 'cpu'
        if kwargs['device'][:3] == 'xpu':
            assert torch.xpu.device_count() > 0, "XPU not available"
            self.ipex_llm_device = kwargs['device']
        else:
            self.ipex_llm_device = 'cpu'
        super().__init__(*args, **kwargs)

        
    def _create_model(
        self,
        pretrained: str,
        revision: Optional[str] = "main",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        trust_remote_code: Optional[bool] = False,
        # arguments used for splitting a model across GPUs naively.
        # only used if `parallelize=True`.
        # (accelerate naive PP (device_map) options)
        parallelize: Optional[bool] = False,
        device_map_option: Optional[str] = "auto",
        max_memory_per_gpu: Optional[Union[int, str]] = None,
        max_cpu_memory: Optional[Union[int, str]] = None,
        offload_folder: Optional[str] = "./offload",
        # PEFT, delta weights and quantization options
        peft: Optional[str] = None,
        delta: Optional[str] = None,
        autogptq: Optional[Union[bool, str]] = False,
        **kwargs,
    ) -> None:
        """
        Initializes an HF or HF-compatible PreTrainedModel from scratch
        inside HFLM, using the kwargs passed into self.__init__().

        Also handles functionality such as AutoGPTQ usage and PEFT wrapping.

        For future similar extensions to AutoGPTQ that are not core to HF's ecosystem,
        (such as PyTorch models that are nearly, but not quite, fully mirroring
        HF's public interface relied on in this HFLM class)
        please consider subclassing HFLM and overriding this and other methods as needed.
        """

        model_kwargs = kwargs if kwargs else {}

        # not supported
        assert not parallelize
        assert not peft
        assert not delta

        model_kwargs.update({"device_map": {"": str(self.device)}})
        cpu_embedding = model_kwargs['cpu_embedding'] if 'cpu_embedding' in model_kwargs.keys() else False

        if model_kwargs['precision'] == 'llb':
            ipex_llm_kwargs = {
                "optimize_model": True,
                "trust_remote_code": trust_remote_code,
                "use_cache": True,
                "cpu_embedding": cpu_embedding
            }  
            self._model = ipex_llm.transformers.AutoModelForCausalLM.load_low_bit(
                pretrained,
                **ipex_llm_kwargs,
            ).eval()
        else:
            ipex_llm_kwargs = {
                "optimize_model": True,
                "load_in_low_bit": model_kwargs['precision'],
                "trust_remote_code": trust_remote_code,
                "use_cache": True,
                "cpu_embedding": cpu_embedding
            }
            self._model = ipex_llm.transformers.AutoModelForCausalLM.from_pretrained(
                pretrained,
                **ipex_llm_kwargs,
            ).eval()

        self._model = self._model.half()
        print("Convert model to half precision")

        self._model = self._model.to(self.ipex_llm_device)
        self._device = torch.device(self.ipex_llm_device)  # override HFLM that changes 'xpu' to 'cpu'

        return None

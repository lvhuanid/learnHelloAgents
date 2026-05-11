# my_llm.py
import os
from typing import Optional
from openai import OpenAI
from hello_agents import HelloAgentsLLM


class MyLLM(HelloAgentsLLM):
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: Optional[str] = "auto",
        **kwargs
    ):
        # 检查provider是否为我们想处理的'modelscope'
        if provider == "modelscope":
            print("正在使用自定义的 ModelScope Provider（通过 OpenAI 兼容模式）")
            
            # 读取 ModelScope 凭证
            api_key = api_key or os.getenv("MODELSCOPE_API_KEY")
            base_url = base_url or "https://api-inference.modelscope.cn/v1/"
            model = model or os.getenv("LLM_MODEL_ID") or "Qwen/Qwen3.5-35B-A3B"
            
            if not api_key:
                raise ValueError("ModelScope API key not found. ...")
            
            # 把 ModelScope 伪装成 OpenAI 兼容的 provider，交给父类处理
            super().__init__(
                model=model,
                api_key=api_key,
                base_url=base_url,
                provider="openai",          # 关键：让父类走 OpenAI 适配器
                **kwargs
            )
        else:
            super().__init__(model=model, api_key=api_key, base_url=base_url, provider=provider, **kwargs)

from langchain_core.embeddings import Embeddings
import requests as _req
import aiohttp as _aio
from typing import List, Union, Dict
from langchain_community.chat_models.zhipuai import _get_jwt_token
import cachetools.func


class ZhipuEmbeddings(Embeddings):
    """Interface for ZhiPu embedding models."""
    _API = "https://open.bigmodel.cn/api/paas/v4/embeddings"
    _MODEL = "embedding-2"
    _API_TOKEN_TTL_SECONDS = 3 * 60
    _CACHE_TTL_SECONDS = _API_TOKEN_TTL_SECONDS - 30

    def __init__(self, api_key: str):
        self.zhipuai_api_key = api_key

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        data = self.__post(text)
        return data["data"][0]["embedding"]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous Embed search docs."""
        return [await self.aembed_query(text) for text in texts]

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronous Embed query text."""
        data = await self.__apost(text)
        return data["data"][0]["embedding"]

    def __post(self, input: str) -> Dict:
        response = _req.post(
            url=ZhipuEmbeddings._API,
            headers=self.__headers(),
            json=ZhipuEmbeddings.__payload(input),
        )
        if not response.ok:
            print("request: ", response.request.body)
            print(response.text)
        response.raise_for_status()
        return response.json()

    async def __apost(self, input: str) -> Dict:
        async with _aio.ClientSession() as session:
            async with session.post(
                url=ZhipuEmbeddings._API,
                headers=self.__headers(),
                data=ZhipuEmbeddings.__payload(input),
            ) as response:
                if not response.ok:
                    print(response.text)
                response.raise_for_status()
                return await response.json()

    def __headers(self) -> Dict[str, str]:
        if self.zhipuai_api_key is None:
            raise ValueError("Did not find zhipuai_api_key.")
        return {
            "Authorization": ZhipuEmbeddings.__jwt(self.zhipuai_api_key),
            "Accept": "application/json",
        }

    @staticmethod
    def __payload(input: Union[str, List[str]]) -> Dict:
        return {
            "input": input,
            "model": ZhipuEmbeddings._MODEL
        }

    @cachetools.func.ttl_cache(maxsize=10, ttl=_CACHE_TTL_SECONDS)
    @staticmethod
    def __jwt(api_key: str):
        return _get_jwt_token(api_key)

from datasets.load import DatasetDict
from dotenv import load_dotenv
import os
from datasets import load_dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
import langchain_community.chat_models.zhipuai as zp
from ragas.testset.generator import TestsetGenerator
from zhipu_embeddings import ZhipuEmbeddings

# that's where API_KEY is stored
load_dotenv()

# loading the V2 dataset
amnesty_qa = load_dataset("explodinggradients/amnesty_qa", "english_v2")
assert type(amnesty_qa) is DatasetDict

# model
zhipuai_key = os.getenv("ZHIPUAI_API_KEY")
assert zhipuai_key is not None
zhipu_model = zp.ChatZhipuAI(
    api_key=zhipuai_key,
    model="glm-4"
)
zhipu_embeddings = ZhipuEmbeddings(api_key=zhipuai_key)

result = evaluate(
    amnesty_qa["eval"],
    llm=zhipu_model,
    embeddings=zhipu_embeddings,
    metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall,
    ],
    raise_exceptions=False
)
result = result.to_pandas()
print(result)

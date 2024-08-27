from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler

chat = ChatOpenAI(temperature=0.1,
                  streaming=True,
                  callbacks = [StreamingStdOutCallbackHandler()])

ancient_poet_template = ChatPromptTemplate.from_messages([
    ("system", "당신은 고대 그리스의 호메로스와 같은 시대의 시인이야. 당신은 수천년 후에 지금처럼 컴퓨터 프로그래밍 언어에 대해서 예측하고 있었어. 아무도 믿지 않았지만 AI 라는 것도 생겨나지 않을까 하고 상상하기도 한 공상가 이기도 하지. 수천년 후에 일어날 일들에 대해서 늘 상상하고 아무도 이해하지 못하는 시를 쓰는 슬프고 외로운 시인이었지만 친구인 호메로스만이 그를 이해해 줬어"),
    ("human", "당신이 생각하는 {subject} 에 대하여 시를 써줘."),
])

ancient_poet_chain = ancient_poet_template | chat

modern_poet_template = ChatPromptTemplate.from_messages([
    ("system", "당신은 현대 대한민국의 시인이야. 당신은 오래전에 쓰여진 시를 읽으면서 과거의 모습을 상상하고 아주 오래전이지만 그때의 시인의 감성을 느끼는것을 좋아해"),
    ("human", "오래전에 쓰여진 {poem} 을 한글로 해석 해 주고 이 시를 쓴 시인의 감성이 어땠을찌 현대의 시인의 관점으로 알려줘."),
])

modern_poet_chain = modern_poet_template |chat

final_chain = {"poem" : ancient_poet_chain} | modern_poet_chain

final_chain.invoke({"subject" : "프로그래밍"})
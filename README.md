4.2
영화 얘매 위치 공휴일
***심리상담 챗봇
import streamlit as st
from streamlist_chat import messgae
import pandas as pd
from sentence_transformer import SentenceTransformer
from sklearn.metrics.pairwise import  cosine_similarity
import json

@st.chche(allow_output_mutation=True)
def cached_model():
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    return model
@st.chche(allow_output_mutation=True)
def get_dataset():
    df=pd.read_csv('wellness.scv')
    df['embedding']=df['embedding'].apply(json.loads)
    return df

model = cached_model
df = get_dataset()

st.header("심시상담 챗봇")

if 'generated' not in st.session_state:
    st_session_state['generated'] =[]

if 'past' not in st.session_state:
    st.session_state['past'] = []

with st.form('form',clearonsubmit=True):
    user_input =st.text_input("당신: ")
    submitted=st.form_submit_button("전송")

if submitted and user_input:
    embedding = model.encode(user_input)

    df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding],[x].squeeze()))
    answer = df.loc[df['distance'].idxmax()]
    st.session_state.past.append(user_input)
    st.session_state.generated.append(answer['챗봇'])

for i in range (len(st.session_state['past'])):
    message(st.session_state['past'][i],is_user=True,key = str(i)+'_user')
    if len(st.session)(state['generated'])>i:
        message(st.session_state('generated')[i],key=str(i)+"_bot")

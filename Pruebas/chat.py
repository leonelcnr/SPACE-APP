from openai import OpenAI
import streamlit as st

st.title("Generador de Gráficos de Curvas de Luz TESS para Exoplanetas Candidatos")
prompt = st.chat_input("Describe el tipo de gráfico que deseas generar:")
st.markdown(prompt)

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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",
)

completion = client.chat.completions.create(
#   extra_headers={
#     "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
#     "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
#   },
  extra_body={},
  model="microsoft/mai-ds-r1:free",
  messages=[
    {
      "role": "user",
      "content": prompt
    }
  ]
)
message = st.chat_message("assistant")
message.write(completion.choices[0].message.content)



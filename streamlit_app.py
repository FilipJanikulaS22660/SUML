import streamlit as st
import time
import streamlit as st
from transformers import pipeline
#import transformers

st.success('Gratulacje! Z powodzeniem uruchomiłeś aplikację')

st.title('Zadanie streamlit SUML 🖥️')

st.header('Przetwarzanie języka naturalnego')

st.write('Aplikacja posiada dwie opcje. Pierwsza pozwala określić wydźwięk emocjonalny tekstu. Druga pozwala na przetłumaczenie angielskiego tekstu na niemiecki.')

st.write('Funkcja tłumaczenie korzysta z modelu google-t5/t5-base udostępnionego na platformie Hugging Face https://huggingface.co/google-t5/t5-base')

st.write('Aby użyć opcji tłumaczenie należy z rozwijanego menu wybrać "Tłumaczenie angielskiego na niemiecki"')

def typewriter(text: str, speed: int):
                tokens = text.split()
                container = st.empty()
                for index in range(len(tokens) + 1):
                    curr_full_text = " ".join(tokens[:index])
                    container.markdown(curr_full_text)
                    time.sleep(1 / speed)

option = st.selectbox(
    "Opcje",
    [
        "Wydźwięk emocjonalny tekstu (eng)",
        "Tłumaczenie angielskiego na niemiecki",
    ],
)

if option == "Wydźwięk emocjonalny tekstu (eng)":
    text = st.text_area(label="Wpisz tekst")
    if text:
        classifier = pipeline("sentiment-analysis")
        answer = classifier(text)
        st.write(answer)
elif option =="Tłumaczenie angielskiego na niemiecki":
    st.image('germ.jpg', caption="image of a building with a flag")
    sentence = st.text_area(label="Wpisz tekst po angielsku")
    if sentence:
        with st.spinner(text='Trwa tłumaczenie...'):

            translator = pipeline("translation_en_to_de", model="t5-base")

            translation = translator(sentence, max_length=1024)
            translated = translation[0]['translation_text']


        if translated == sentence:
            st.error('Błąd, wiadomość nie jest w języku angielskim.')
        else:
            text = translated
            speed = 15
            st.success('Tłumaczenie gotowe!')
            st.write("Tłumaczenie:")
            typewriter(text=text, speed=speed)
            st.balloons()

st.header('Filip Janikula s22660')

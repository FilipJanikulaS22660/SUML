import streamlit as st
import time
import streamlit as st
from transformers import pipeline
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
# zaczynamy od zaimportowania bibliotek

st.success('Gratulacje! Z powodzeniem uruchomiłeś aplikację')

st.title('Zadanie streamlit SUML 🖥️')

st.header('Przetwarzanie języka naturalnego')

st.write('Aplikacja posiada dwie opcje. Pierwsza pozwala określić wydźwięk emocjonalny tekstu. Druga pozwala na przetłumaczenie angielskiego tekstu na niemiecki.')

st.write('Funkcja tłumaczenie korzysta z modelu Llama2-13b-Language-translate udostępnionego na platformie Hugging Face https://huggingface.co/SnypzZz/Llama2-13b-Language-translate')

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
            model = MBartForConditionalGeneration.from_pretrained("SnypzZz/Llama2-13b-Language-translate")
            tokenizer = MBart50TokenizerFast.from_pretrained("SnypzZz/Llama2-13b-Language-translate", src_lang="en_XX")

            model_inputs = tokenizer(sentence, return_tensors="pt")
            generated_tokens = model.generate(
                **model_inputs,
                forced_bos_token_id=tokenizer.lang_code_to_id["de_DE"]
            )

        translated = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

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
        

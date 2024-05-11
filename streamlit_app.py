import streamlit as st
import time
# zaczynamy od zaimportowania bibliotek

st.success('Gratulacje! Z powodzeniem uruchomiłeś aplikację')
# streamlit jest wykorzystywany do tworzenia aplikacji
# z tego powodu dobrą praktyką jest informowanie użytkownika o postępie, błędach, etc.

# Inne przykłady do wypróbowania:
 # animowane balony ;)
# st.error('Błąd!') # wyświetla informację o błędzie
# st.warning('Ostrzeżenie, działa, ale chyba tak sobie...')
# st.info('Informacja...')
# st.success('Udało się!')

# st.spinner()
# with st.spinner(text='Pracuję...'):
    # time.sleep(2)
    # st.success('Done')
# możemy dzięki temu "ukryć" późniejsze ładowanie aplikacji

st.title('Zadanie streamlit SUML 🖥️')
# title, jak sama nazwa wskazuje, używamy do wyświetlenia tytułu naszej aplikacji


# header to jeden z podtytułów wykorzystywnaych w Streamlit

# subheader to jeden z podtytułów wykorzystywnaych w Streamlit

#st.text('To przykładowa aplikacja z wykorzystaniem Streamlit')
# text używamy do wyświetlenia dowolnego tekstu. Można korzystać z polskich znaków.

#st.write('Streamlit jest biblioteką pozwalającą na uruchomienie modeli uczenia maszynowego.')
# write używamy również do wyświetlenia tekstu, różnica polega na formatowaniu.

#st.code("st.write()", language='python')
# code może nam się czasami przydać, jeżeli chcielibyśmy pokazać np. klientowi fragment kodu, który wykorzystujemy w aplikacji

#with st.echo():
#    st.write("Echo")
# możemy też to zrobić prościej używając echo - pokazujemy kod i równocześnie go wykonujemy

#df = pd.read_csv("DSP_4.csv", sep = ';')
#st.dataframe(df)
# musimy tylko pamiętać o właściwym określeniu separatora (w tym wypadku to średnik)
# masz problem z otworzeniem pliku? sprawdź w jakim katalogu pracujesz i dodaj tam plik (albo co bardziej korzystne - zmień katalog pracy)
# os.getcwd() # pokaż bieżący katalog
# os.chdir("") # zmiana katalogu

st.header('Przetwarzanie języka naturalnego')

st.write('Aplikacja posiada dwie opcje. Pierwsza pozwala określić wydźwięk emocjonalny tekstu. Druga pozwala na przetłumaczenie angielskiego tekstu na niemiecki.')

st.write('Funkcja tłumaczenie korzysta z modelu Llama2-13b-Language-translate udostępnionego na platformie Hugging Face https://huggingface.co/SnypzZz/Llama2-13b-Language-translate')

st.write('Aby użyć opcji tłumaczenie należy z rozwijanego menu wybrać "Tłumaczenie angielskiego na niemiecki"')

import streamlit as st
from transformers import pipeline
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

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
        

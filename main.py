import numpy as np
import streamlit as st 
from stqdm import stqdm 
from model_in_cache import mod 
from streamlit_styles.header_styles import styles
from streamlit_mods.links import links
import matplotlib.pyplot as plt
from streamlit_mods.sidebar import sidebar
from model_in_cache.mod import load_all_models, read_data, read_glove_vecs
from pred.pred import NMT, format_date, NER, built_entities, ner_header, nmt_header, question_duplicate
from pred.pred import nmt_attention_header, sa, huggingface, qa_t5
from built.siamense import siamense_prediction
from utils.utils import sentences_to_indices, label_to_emoji, custom_ner, custum_sa

import time
import torch


def my_app():
    """
    st.set_page_config(
    page_title="FloraFlow",
    page_icon="🌱"
    )
    """
    yolo_logo = './images/ocr.png'
    git_page  = links('git_page')

    st.set_page_config(
            page_title="My Streamlit App",
            page_icon=":chart_with_upwards_trend:",
            initial_sidebar_state="expanded",
            layout="centered",
        )
    
    M = 10000
    st.image(plt.imread("./images/R.jpeg"))
    #st.markdown(f'<a href="{git_page}" target="_blank"><img src="{yolo_logo}" width="450" height="200"></a>', unsafe_allow_html=True)
    
    # Définir le style CSS personnalisé
    custom_css = styles()

    # Appliquer le style CSS personnalisé
    st.write('<style>{}</style>'.format(custom_css), unsafe_allow_html=True)

    # buttom configuration
    st.write(
        f'<style>div.stButton > button{{background-color: white; color: black; \
            padding: 10px 20px, text-align: center;\
            display: inline-block;\
            font-size: 16px;\
            border-radius: 50px;\
            background-image: linear-gradient(to bottom, red, darkorange, orange);\
            font-weight: bolder}} </style>',
        unsafe_allow_html=True
    )

    # selectbox configuration
    custom_style = "<style>\
                    div[data-baseweb='select'] { background-color: #f2f2f2; \
                    background-image: linear-gradient(to bottom, darkgreen, lime, gray);\
                    border: 1px solid #ccc;\
                    border-radius: 5px;\
                    padding: 5px;\
                    font-size: 16px;\
                    font-family: Arial, sans-serif;\
                    max-height: 50px;\
                    overflow-y: auto;\
                    }\
                    </style>"
    
    st.write(custom_style, unsafe_allow_html=True)
    #st.markdown(custom_style, unsafe_allow_html=True)
    # Insérer du HTML personnalisé pour personnaliser la case à cocher
    st.markdown("""
        <style>
            /* Personnalisation de la case à cocher */
            .custom-checkbox-label {
                display: inline-block;
                cursor: pointer;
                position: relative;
                padding-left: 25px;
            }

            .custom-checkbox-label::before {
                content: "";
                display: inline-block;
                position: absolute;
                width: 18px;
                height: 18px;
                left: 0;
                top: 0;
                border: 1px solid #000;
                background-color: #fff;
            }

            input[type="checkbox"] {
                display: none;
            }

            input[type="checkbox"]:checked + .custom-checkbox-label::before {
                background-color: #007BFF;
            }
        </style>
    """, unsafe_allow_html=True)

    # Utiliser le style CSS personnalisé pour afficher du texte en surbrillance
    st.write('<h1 class="custom-text">Natural Language Processing with TensorFlow & PyTorch</h1>', unsafe_allow_html=True)

    dataset, human_vocab, machine_vocab, inv_machine_vocab = read_data(m=M)
    all_models, loaded_sentence_vectorizer = load_all_models(machine_vocab)
    words_to_index  = read_glove_vecs()
    #sentinels       = get_sentinels()

    contain_feedback    = sidebar(streamlit=st)
    #st.write('<h1 class="custom-text-under"></h1>', unsafe_allow_html=True)
    
    if contain_feedback == "Neural Machine Translation":
        fmt_date, sep, data = nmt_header(st=st)

        #EXAMPLES = ['3 May 1979', '5 April 09', '21th of August 2016', 'Tue 10 Jul 2007', 
        #            'Saturday May 9 2018', 'March 3 2001', 'March 3rd 2001', '1 March 2001']
        if st.button("run model"):
            if data:
                EXAMPLES = data
                model = all_models['NMT']
                prediction, True_value = NMT(human_vocab, inv_machine_vocab, model, EXAMPLES)
                prediction = format_date(prediction, sep=sep, format=fmt_date)
                result = {"Human data" : True_value, "Machine Translation" : prediction}
                st.json( result )
            else:
                st.warning("⚠️ empty text")

    if contain_feedback == 'Named Entity Recognition':
        sentence = ner_header(st=st)
        #sentence = "Peter Parker , the White House director of trade and manufacturing policy of U.S , said in an interview on Sunday morning that the White House was working to prepare for the possibility of a second wave of the coronavirus in the fall , though he said it wouldn ’t necessarily come"
        if isinstance(sentence, str):
            if st.button("run model", disabled=False if sentence else True):
                if sentence:
                    model = all_models['NER']
                    prediction = NER(model=model, sentence=sentence, sentence_vectorizer=loaded_sentence_vectorizer)
                    prediction = built_entities(prediction)
                    st.json(prediction)
        else:
            if st.button("run model", disabled=False if sentence.shape != () else True):
                s = st.empty()
                progress_text   = "Operation in progress. Please wait."
                my_bar          = st.progress(0, text=progress_text)
                sentence        = sentence[:10]
                size            = sentence.shape[0]

                for i, string in stqdm(enumerate(sentence), backend=False, frontend=True):
                    model = all_models['NER']
                    prediction = NER(model=model, sentence=string, sentence_vectorizer=loaded_sentence_vectorizer)

                    time.sleep(0.01)
                    if size < 100:
                        j = (i + 1) * int(100 / size)
                        j =  j if j < 100 else 100
                        my_bar.progress(j, text=progress_text)
                    else:
                        if i > (size / 100):
                            my_bar.progress(i + 1, text=progress_text)

                prediction = built_entities(prediction)
                st.json(prediction)

    if contain_feedback == "Question Duplicates":
        t, q1, q2 = question_duplicate(st=st)

        if st.button('run model'):
            if q1 and q2:
                model = all_models['siamense']
                result = siamense_prediction(question1=q1, question2=q2, threshold=t, model=model)

                st.json(result)
            
            else:
                if not q1:
                    st.warning("⚠️ question 1 is empty")
                if not q2:
                    st.warning("⚠️ question 2 is empty")
    
    if contain_feedback == "Neural Machine Translation11":
        (sample, temp, sim, lang, text) = nmt_attention_header(st=st)

        if text:
            if st.button('Translate'):
                pass 
        else:
            st.warning("⚠️ empty text")
    
    if contain_feedback == 'Sentiment Analysis':
        type_sa, inp = sa(st=st)

        if st.button("run model"):
            if inp:
                if type_sa == "Emojyfy":
                    model = all_models['emojify']
                
                    inp = np.array([x.rstrip().lstrip() for x in inp.split(';') if x] )       
                    inp_indices = sentences_to_indices(inp, words_to_index)
                    pred = model.predict(inp_indices)
                    data = {}

                    for i in range(len(pred)):
                        num = np.argmax(pred[i], axis=-1)
                        data[inp[i]] = label_to_emoji(num).strip()
                        
                    if data:
                        st.json(data)
            
            else:
                st.warning("⚠️ empty text")

    if contain_feedback == 'BERT & HuggingFace':
        (task, fine_tuning, context, questions) = huggingface(st=st)

        if task == "Quetion Answering":
            if st.button('run model'):
                if questions:
                    if fine_tuning is False:
                        model = all_models['HF_QA']
                        result = model(question=questions, context=context)
                    else: 
                        model = all_models['HF_QA_FT']
                        tokenizer = all_models['HF_QA_T']
                        result = {}
                        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
                        for question in questions:
                            inputs = tokenizer.encode_plus(question, context, return_tensors="pt")
                            input_ids = inputs["input_ids"].tolist()[0]

                            inputs.to(device)
                            text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
                            answer_model = model(**inputs)
                            start_logits = answer_model['start_logits'].cpu().detach().numpy()
                            answer_start = np.argmax(start_logits)  
                            end_logits = answer_model['end_logits'].cpu().detach().numpy()
                            # Get the most likely beginning of answer with the argmax of the score
                            answer_end = np.argmax(end_logits) + 1  # Get the most likely end of answer with the argmax of the score
                            answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

                            result[question] = answer
                            
                    st.json(result)
                else:
                    if not context:
                        st.warning("⚠️ empty context")
                    else:
                        st.warning("⚠️ empty question")
        
        elif task == 'Named Entity Recognition':
            if st.button('run model'):
                if context:
                    model = all_models['HF_NER']
                    tokenizer = all_models['HF_NER_T']
                    # Exemple d'utilisation
                    result = {"corpus" : [], "entities" : []}
                    for text in context:
                        entities = custom_ner(text, tokenizer, model)
                        if entities:
                            result['corpus'].append(text)
                            result['entities'].append(entities)

                    st.json(result)
                else:
                    st.warning("⚠️ empty context")
        
        elif task == "Sentiment Analysis":
            if st.button('run model'):
                if context:
                    model = all_models['HF_SA']
                    tokenizer = all_models['HF_SA_T']
                    result = {"corpus" : [], "sentiment" : []}

                    for text in context:
                        label = custum_sa(text, tokenizer, model)
                        result['corpus'].append(text)
                        result['sentiment'].append(label)
                     
                    st.json(result)
                else:
                    st.warning("⚠️ empty context")
        
        else:  pass

    if contain_feedback == "Question Answering With T5":
        context, questions, q = qa_t5(st=st)

        if st.button('run model'):
            if context:
                if questions:

                    
                    #model = all_models["QA_with_T5"]
                    #result = decode(questions=questions, model=model, sentinels=sentinels, q=q)
                    #result['context'] = context
                    #st.json(result)
                    

                    model = all_models['HF_QA_FT']
                    tokenizer = all_models['HF_QA_T']
                    result = {"context" : context, "questions" : [], "answers" : [], "ids" : []}
                    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            
                    for i, question in enumerate([x.strip() for x in q.replace("\n", '').replace("\t", '').split(";")]):
                        inputs = tokenizer.encode_plus(question, context, return_tensors="pt")
                        input_ids = inputs["input_ids"].tolist()[0]

                        inputs.to(device)
                        text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
                        answer_model = model(**inputs)
                        start_logits = answer_model['start_logits'].cpu().detach().numpy()
                        answer_start = np.argmax(start_logits)  
                        end_logits = answer_model['end_logits'].cpu().detach().numpy()
                        # Get the most likely beginning of answer with the argmax of the score
                        answer_end = np.argmax(end_logits) + 1  # Get the most likely end of answer with the argmax of the score
                        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

                        result["questions"].append(question)
                        result['answers'].append(answer)
                        result['ids'].append(i)

                    st.json(result)
                else:
                    st.warning("⚠️ empty questions")
            else:
                st.warning("⚠️ empty context")

if __name__ == "__main__":
    my_app()
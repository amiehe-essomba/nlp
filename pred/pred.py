import numpy as np
import json 
from modules.nmt import string_to_int
from keras.utils import to_categorical
import streamlit 
import tensorflow as tf
import matplotlib.pyplot as plt
from streamlit_styles.body_styles import b_styles
from utils.utils import preprocess_image, online_link

def read_img(st, uploaded_file):
    types               = ["jpg", "jpeg", "png", "gif", "webp", "mp4", "mov", "avi", "mkv"]
    image, image_data   = (None, None)

    for file in uploaded_file:
        file_extension = file.name.split(".")[-1].lower()

        if file_extension in types:
            if file_extension in ["jpg", "jpeg", "png", "gif", "webp"]:
                try:
                    image, image_data = preprocess_image(img_path=file)
                    break
                except (FileNotFoundError, FileExistsError) : 
                    st.white("File loading error")
                    process = False
                    break
            else: st.write(f"⚠️ The file {file.name} is not an image")
    
    return image, image_data

def NMT(human_vocab : dict, inv_machine_vocab : dict, model, ss):
    Tx, n_s, Ty     = 30, 64, 10
    s00, c00        = np.zeros((1, n_s)), np.zeros((1, n_s))
    outputs         = []
    sources         = []
    EXAMPLES        = ss#get_string()

    for example in EXAMPLES:
        source = string_to_int(example, Tx, human_vocab)
        source = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), source))).swapaxes(0,1)
        source = np.swapaxes(source, 0, 1)
        source = np.expand_dims(source, axis=0)
        prediction = model.predict([source, s00, c00])
        prediction = np.argmax(prediction, axis = -1)
        output = [inv_machine_vocab[int(i)] for i in prediction]

        outputs.append(''.join(output))
        sources.append(example)
       
    return outputs, sources

def format_date(pred_output : list, format : str = "yy-mm-dd", sep : str = '-'):
    
    if pred_output:
        if format == "yy-mm-dd": 
            for i, string in enumerate(pred_output):
                string = string.replace("-", sep)
                pred_output[i] = string
        elif format == 'dd-mm-yy':
            for i, string in enumerate(pred_output):
                y, m, d = string.split('-')
                _ = [d, sep, m, sep, y]
                string = ''.join(_)
                pred_output[i] = string
        elif format == "yy-dd-mm":
            for i, string in enumerate(pred_output):
                y, m, d = string.split('-')
                _ = [y, sep, d, sep, m]
                string = ''.join(_)
                pred_output[i] = string
        
        elif format == "mm-dd-yy":
            for i, string in enumerate(pred_output):
                y, m, d = string.split('-')
                _ = [m, sep, d, sep, y]
                string = ''.join(_)
                pred_output[i] = string
        
        elif format == "mm-yy":
            for i, string in enumerate(pred_output):
                y, m, d = string.split('-')
                _ = [m, sep, y]
                string = ''.join(_)
                pred_output[i] = string
        
        elif format == "dd-mm":
            for i, string in enumerate(pred_output):
                y, m, d = string.split('-')
                _ = [d, sep, m]
                string = ''.join(_)
                pred_output[i] = string
    
    return pred_output

def convert_format_for_ner() -> dict:
    entities = {
                "B-geo" : "geographical entity",
                "B-org" : "organization",
                "B-per" : "person",
                "B-gpe" : "geopolitical entity",
                "B-tim" : "time indicator",
                "B-art" : "artifact",
                "B-eve" : "event",
                "B-nat" : "natural phenomenon",
                "I-geo" : "geographical entity",
                "I-org" : "organization",
                "I-per" : "person",
                "I-gpe" : "geopolitical entity",
                "I-tim" : "time indicator",
                "I-art" : "artifact",
                "I-eve" : "event",
                "I-nat" : "natural phenomenon",
                "O"   : "filler word" 
            }
    

    return entities

def NER(sentence, model, sentence_vectorizer):

    
    """
    Predict NER labels for a given sentence using a trained model.

    Parameters:
    sentence (str): Input sentence.
    model (tf.keras.Model): Trained NER model.
    sentence_vectorizer (tf.keras.layers.TextVectorization): Sentence vectorization layer.
    tag_map (dict): Dictionary mapping tag IDs to labels.

    Returns:
    predictions (list): Predicted NER labels for the sentence.

    """

    tag_map = {
                'B-art': 0, 'B-eve': 1, 'B-geo': 2, 'B-gpe': 3, 
                'B-nat': 4, 'B-org': 5, 'B-per': 6, 'B-tim': 7, 
                'I-art': 8, 'I-eve': 9, 'I-geo': 10, 'I-gpe': 11, 
                'I-nat': 12, 'I-org': 13, 'I-per': 14, 'I-tim': 15, 
                'O': 16
                }

    # Convert the sentence into ids
    sentence_vectorized = sentence_vectorizer(sentence)
    # Expand its dimension to make it appropriate to pass to the model
    sentence_vectorized = tf.expand_dims(sentence_vectorized, axis=0)
    # Get the model output
    output = model.predict(sentence_vectorized)
    # Get the predicted labels for each token, using argmax function and specifying the correct axis to perform the argmax
    outputs = tf.math.argmax(output, axis = -1)
    # Next line is just to adjust outputs dimension. Since this function expects only one input to get a prediction, outputs will be something like [[1,2,3]]
    # so to avoid heavy notation below, let's transform it into [1,2,3]
    outputs = tf.squeeze(outputs)
    # Get a list of all keys, remember that the tag_map was built in a way that each label id matches its index in a list
    labels = list(tag_map.keys()) 
    pred = [] 
    # Iterating over every predicted token in outputs list
   
    if outputs.shape != ():
        for tag_idx in outputs:
            pred_label = labels[tag_idx]
            pred.append(pred_label)
        
        sentence = sentence.split()

    items = []
    for x, y in zip(sentence, pred):
        if y != 'O':
            items.append([x, y])
    return items

def built_entities(items):    
    format_ner          = convert_format_for_ner()
    unique_values       = list( set( list( format_ner.values() ) ) )
    unique              = {unique_values[i] : set() if unique_values[i] != "person" else {
                                                    'Firt Name' : set(),
                                                    "Second Name" : set()
                                            } for i in range(len(unique_values)) }
    is_found            = False 

    if items:
        is_found = True 
        for i, value in enumerate(items):
            name, key   = value 

            if isinstance(name, str): pass 
            else: name = name.decode('utf-8')
            
            if key not in ['B-per', 'I-per']:
                key         = format_ner[key]
                j           = unique_values.index(key)
                unique[unique_values[j]].add(str(name))
            else:
                key1         = format_ner[key]
                j           = unique_values.index(key1)
                if key == 'B-per':
                    unique[unique_values[j]]['Firt Name'].add(str(name))
                else:
                    unique[unique_values[j]]["Second Name"].add(str(name))
    else: pass

    #unique = json.dumps(unique, skipkeys=True, indent=4)

    return unique, is_found

def ner_header(st : streamlit):
    def load_data(file_path):
        import numpy as np 

        with open(file_path,'rb') as file:
            data = np.array([line.strip() for line in file.read()])
        return data

    st.write('<style>{}</style>'.format(b_styles()), unsafe_allow_html=True)
    st.write(f'<h1 class="header-text"> Welcome in Named Entity Recognition Section</h1>', unsafe_allow_html=True)
    for i in range(2):
        st.write('')

    col1, col2, col3 = st.columns(3)
    result = ""
    data   = ""
    with col1:
        result = st.selectbox(label=" ", options=("Load a text", "Write a text"), index=None, placeholder='Select method')
        st.write('status', True if result else False)

    if result:
        if result == 'Load a text':
            r = st.file_uploader("", type="txt", accept_multiple_files=True)
            for file in r:
                file_extension = file.name.split(".")[-1].lower()
                if file_extension == 'txt':
                    try:
                        data = np.array([line.strip() for line in file.readlines()])
                    except Exception:
                        st.warning(f'cannot upload {file.name}')
                else:
                    st.warning(f'{file.name} is not a text file')
        if result == 'Write a text':
            text = 'Peter Parker, the White House director of trade .....'
            data = st.text_area('Write you text here please', placeholder=text)
            if data:
                try:
                    data = data.rstrip().lstrip().replace('\n', '').replace('\t', '')
                except Exception: pass
            
    return data

def nmt_header(st : streamlit):

    st.write('<style>{}</style>'.format(b_styles()), unsafe_allow_html=True)
    st.write(f'<h1 class="header-text">Welcome in Human Date Translation Section </h1>', unsafe_allow_html=True)
    st.write(f'<h1 class="header-text-under">NOTE : The model was trained with years less than 2024.\
             You can put several dates separated by the comma.</h1>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    data   = ""
    ner    = False 

    with col1:
        format_date = st.selectbox(label=" ", 
                              options=("dd-mm-dd",  "yy-dd-mm", "mm-yy", "mm-dd",
                                        'yy-mm-dd', "mm-dd-yy", "yy-mm", "dd-mm"
                                        ), 
                              index=None, placeholder='Select format')
        st.write('status', True if format_date else False)
    
    with col2:
        sep = st.selectbox(label=" ", 
                              options=("-", "/", "_", ".", "--", '//', ";", ":"), 
                              index=None, placeholder='Select separator')
        st.write('status', True if sep else False)
    
    with col3:
        ner = st.checkbox(label="Analyze TEXT with NER", disabled=True)
        st.write('status', True if ner else False)

    text = '21th of August 2016'
    data = st.text_area('Write you text here please', placeholder=text)
    
    format_date = "dd-mm-dd" if not format_date else format_date
    sep         = '-' if not sep else sep

    try:
        if data:
            data        = data.strip().lstrip().replace('\n', '').replace('\t', '')
            data        = [x.strip().lstrip() for x in data.split(',')]
        else: pass
    except Exception: pass 

    return format_date, sep, data

def question_duplicate(st : streamlit):

    st.write('<style>{}</style>'.format(b_styles()), unsafe_allow_html=True)
    st.write(f'<h1 class="header-text">Welcome in Question Duplicates Section </h1>', unsafe_allow_html=True)
    st.write(f'<h1 class="header-text-under">NOTE : Write question1 firstly to activate question2.</h1>', unsafe_allow_html=True)

    col2, col3  = st.columns(2)
    q1, q2      = "", ""
    locked      = True 
    threshold   = 0.7
    
    with col2:
        q1 = st.text_area("question1", placeholder="writre your question1")
        st.write('status', True if q1 else False)
        if q1: locked = False
    
    with col3:
        q2 = st.text_area("question2", placeholder="writre your question2", disabled=locked)
        st.write('status', True if q2 else False)
    
    if q2:
        threshold = st.slider('threshold', min_value=0.1, max_value=1.0, step=0.1, value=0.7)
        st.write('status', True if threshold else False)

    if q1:
        try: q1 = q1.rstrip().lstrip().replace('\n', '').replace('\t', '')
        except Exception:pass
    
    if q2:
        try: q2 = q2.rstrip().lstrip().replace('\n', '').replace('\t', '')
        except Exception:pass

    return threshold, q1, q2

def nmt_attention_header(st : streamlit):

    st.write('<style>{}</style>'.format(b_styles()), unsafe_allow_html=True)
    st.write(f'<h1 class="header-text">Welcome in Neural Machine Translation Section </h1>', unsafe_allow_html=True)
    st.write(f'<h1 class="header-text-under">English-to-Portuguese neural machine translation (NMT)</h1>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        sample = st.slider('samples', min_value=1, max_value=30, value=10, step=1)
        st.write('state', True if sample else False)
    
    with c2:
        temp = st.slider('temperature', min_value=0.1, max_value=1.0, step=0.1, value=0.6)
        st.write('state', True if temp else False)
    
    with c3:
        sim = st.selectbox('similarity function', options=("jaccard", "rounge-n"), index=0)
        st.write('state', True if sim else False)
    
    with c4:
        lang = st.selectbox('Translation', options=("eng-to-french", 'french-to-eng', 'eng-to-por', 'por-to-eng'), index=0)
        st.write('state', True if lang else False)

    if sample and temp and sim and lang:
        text = st.text_area('Text', placeholder=lang)


    return (sample, temp, sim, lang, text)

def sa(st : streamlit):

    st.write('<style>{}</style>'.format(b_styles()), unsafe_allow_html=True)
    st.write(f'<h1 class="header-text">Welcome in Sentiment Analysis Section </h1>', unsafe_allow_html=True)
    st.write(f'<h1 class="header-text-under">NOTE : You can put several texts separated by the semicolon.</h1>', unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        type_sa = st.selectbox('Type of sentiments', ('Emojyfy', "Emojyfy"), index=0, disabled=True)
    with c2:
        p = 'I love you = ❤️\nI hate you = 😞\nI want to play game = ⚾\nI injoyed eating = 😄\nI want to eat = 🍴'
        st.text_area("Examples", disabled=True, placeholder=p)

    inp = st.text_area('TEXT', placeholder="write your text here please")

    if inp:
        try: inp = inp.rstrip().lstrip().replace('\n', '').replace('\t', '')
        except Exception: pass 

    return type_sa, inp
    
def huggingface(st : streamlit):
    st.write('<style>{}</style>'.format(b_styles()), unsafe_allow_html=True)
    st.write(f'<h1 class="header-text">Welcome in BERT & HuggingFace Section </h1>', unsafe_allow_html=True)
    st.write(f'<h1 class="header-text-under">NOTE : You can put several texts separated by the semicolon.</h1>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        task = st.selectbox('HuggingFace Task', ('Quetion Answering', 'Sentiment Analysis'), index=0) #'Named Entity Recognition',
        st.write('state', True if task else False)
    
    with c2:
        disable = False 
        if task == "Quetion Answering": pass 
        else: disable = True 
        fine_tuning = st.checkbox("Use Fine-Tuning", value=False, disabled=disable)

    question = ""
    
    if task == "Quetion Answering":
        context = st.text_area('Context', placeholder="put your context here")

        if context:
            question = st.text_area('Questions', placeholder="put your questions here")
            
            if question:
                try:
                    question = [q.rstrip().lstrip() for q in question.replace('\n', '').replace('\t', '').split(";")]
                except Exception: pass 

        return (task, fine_tuning, context, question)
    
    elif task  in ["Named Entity Recognition", 'Sentiment Analysis']:
        context = st.text_area('Context', placeholder="put your context here")
        try:
            context = [q.rstrip().lstrip() for q in context.replace('\n', '').replace('\t', '').split(";")]
        except Exception: pass 

        return task, None, context, None
    
    else:
        col1, col2 = st.columns(2)

        with col1:
            label_select = st.selectbox('Local or Online File', options=('Local', 'Online'), index=None)
            st.write('state', True if label_select else False)
        
        if label_select == 'Local':
            uploaded_file = st.file_uploader("upload local image or video", 
                                                type=["jpg", "jpeg", "png", "gif", "webp", "mp4", "mov", "avi", "mkv"],
                                                accept_multiple_files=False
                                                )
        elif label_select == "Online":
            uploaded_file = st.text_input('put your url here', placeholder="https://....")
        else: uploaded_file =None

        if uploaded_file:
            if label_select == 'Local':
                image, image_data = read_img(st, uploaded_file=uploaded_file)
            else:
                image, image_data = online_link(st=st, url=uploaded_file)
                st.image(image=image)

            return uploaded_file, image, image_data, None
        
        return None, None, None, None 

def qa_t5(st:streamlit):
    st.write('<style>{}</style>'.format(b_styles()), unsafe_allow_html=True)
    st.write(f'<h1 class="header-text">Welcome in Question Answering With T5 Section </h1>', unsafe_allow_html=True)
    st.image(plt.imread('./images/qa.png'))
    st.write(f'<h1 class="header-text-under">NOTE : You can write saveral questions separated by the semicolon.</h1>', unsafe_allow_html=True)
    
    context  = st.text_area('Context', placeholder="put your Context here")
    question = st.text_area('Questions', placeholder="put your questions here")
    q = question 

    if context:
        try: context = context.strip().replace('\n', '').replace('\t', '')
        except Exception: pass 

    if question: 
        if context:
            question = [f"question: {q.strip()}  context: {context}" for q in question.replace('\n', '').replace('\t', '').split(";") if q]
    
    return context, question, q

    

    return question 
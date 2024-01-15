import numpy as np
import pandas as pd 

import emoji

emoji_dictionary = {"0": "\u2764\uFE0F",    # :heart: prints a black instead of red heart depending on the font
                    "1": ":baseball:",
                    "2": ":smile:",
                    "3": ":disappointed:",
                    "4": ":fork_and_knife:"}

def label_to_emoji(label):
    """
    Converts a label (int or string) into the corresponding emoji code (string) ready to be printed
    """
    return emoji.emojize(emoji_dictionary[str(label)], language='alias')


def sentences_to_indices(X, word_to_index, max_len=10):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    The output shape should be such that it can be given to `Embedding()` (described in Figure 4). 
    
    Arguments:
    X -- array of sentences (strings), of shape (m,)
    word_to_index -- a dictionary containing the each word mapped to its index
    max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this. 
    
    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """
    
    m = X.shape[0]                                   # number of training examples
    
    ### START CODE HERE ###
    # Initialize X_indices as a numpy matrix of zeros and the correct shape (≈ 1 line)
    X_indices = np.zeros((m, max_len))

    for i in range(m):                               # loop over training examples
        # Convert the ith training sentence in lower case and split is into words. You should get a list of words.
        sentence_words = [word.lower() for word in X[i].split()]
        
        # Initialize j to 0
        j = 0
        
        # Loop over the words of sentence_words
        for w in sentence_words:
            # if w exists in the word_to_index dictionary
            if str(w).encode() in list( word_to_index.keys()):
                # Set the (i,j)th entry of X_indices to the index of the correct word.
                X_indices[i, j] = word_to_index[str(w).encode()]
                # Increment j to j + 1
                j = j + 1
            
    ### END CODE HERE ###
    
    return X_indices


def preprocess_image(img_path,  done : bool = False):
    from PIL import Image 

    #image_type = imghdr.what(img_path)
    if done is False : image           = Image.open(img_path)
    else: image = img_path

    shape           = np.array(image).shape[:-1]
    resized_image   = image.resize(tuple(reversed(shape)), Image.BICUBIC)
    image_data      = np.array(resized_image, dtype='float32')
    image_data     /= 255.
    # Add batch dimension.
    image_data      = np.expand_dims(image_data, axis=0) 

    return image, image_data


def online_link(st, url : str = "", show_image : bool = True):
    from skimage.transform import resize

    image, image_data, shape, error = None, None, None, None
    # Vérifie si le champ de saisie n'est pas vide
    if url:
        # Affiche le lien hypertexte
        st.markdown(f"You enter this link : [{url}]({url})")
        image, image_data, shape, error = url_img_read(url=url)

        if error is None:
            if show_image is True:
                st.header(f"image 0, shape = {shape}")
                img_array = resize(np.array(image), output_shape=shape[:-1])
                st.image(img_array, use_column_width=True)
                st.markdown(f"file successfully upladed...")
            else: pass
        else: pass#st.markdown(f"{error}")
    else: st.write(f"⚠️ {error}")

    return image, image_data

def url_img_read( url : str):
    from PIL import Image
    import requests
    from io import BytesIO 

    image, image_data, error, shape = None, None, None, None
    # Replace 'url' with the URL of the image you want to read

    try:
        response = requests.get(url)
        # Check if the request was successful
        if response.status_code == 200:
            # Read the image from the response content
            image_data = BytesIO(response.content)
            #Image.open(image_data)
            image = Image.open(image_data)

            shape = np.array( [x for x in image.size] )
            if (shape < np.array([6000, 6000]) ).all() : 
                image, image_data, shape = preprocess_image(img_path=image, done=True)
            else:
                error = 'image size out of range (6000, 6000)'
        else:
            error = f"Failed to retrieve image. Status code: {response.status_code}"
    except Exception as e:
        error = f"An error occurred: {str(e)}"

    return image, image_data, shape,  error

def custom_ner(text, tokenizer, model):
    import torch 

    # Tokenisation
    tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(text)))
    inputs = tokenizer.encode(text, return_tensors="pt")

    # Prédiction
    with torch.no_grad():
        outputs = model(inputs).logits

    # Obtention des prédictions pour chaque token
    predictions = torch.argmax(outputs, dim=2)

    # Récupération des entités nommées
    entities = []
    current_entity = {"word": "", "label": ""}

    for token, prediction in zip(tokens, predictions[0].tolist()):
        label = model.config.id2label[prediction]
        
        if label.startswith("B-"):
            if current_entity["word"]:
                entities.append(current_entity)
            current_entity = {"word": token, "label": label[2:]}
        elif label.startswith("I-"):
            current_entity["word"] += " " + token
            current_entity["label"] = label[2:]
        else:
            if current_entity["word"]:
                entities.append(current_entity)
            current_entity = {"word": "", "label": ""}
    
    return entities

def custum_sa(text, tokenizer, model):
    import torch 

    tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(text)))
    inputs = tokenizer.encode(text, return_tensors="pt")

    with torch.no_grad():
        logits = model(inputs).logits

    predicted_class_id = logits.argmax().item()
    label = model.config.id2label[predicted_class_id]

    return label 


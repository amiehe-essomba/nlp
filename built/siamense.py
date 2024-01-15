import tensorflow as tf 

def Siamese(text_vectorizer, vocab_size=36224, d_feature=128):
    """Returns a Siamese model.

    Args:
        text_vectorizer (TextVectorization): TextVectorization instance, already adapted to your training data.
        vocab_size (int, optional): Length of the vocabulary. Defaults to 56400.
        d_model (int, optional): Depth of the model. Defaults to 128.
        
    Returns:
        tf.model.Model: A Siamese model. 
    
    """
    ### START CODE HERE ###

    branch = tf.keras.models.Sequential(name='sequential') 
    # Add the text_vectorizer layer. This is the text_vectorizer you instantiated and trained before 
    branch.add(text_vectorizer)
    # Add the Embedding layer. Remember to call it 'embedding' using the parameter `name`
    branch.add(tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=d_feature, input_length=None, name="embeding"))
    # Add the LSTM layer, recall from W2 that you want to the LSTM layer to return sequences, ot just one value. 
    # Remember to call it 'LSTM' using the parameter `name`
    branch.add(tf.keras.layers.LSTM(units=d_feature, return_sequences=True, name="lstm"))
    # Add the GlobalAveragePooling1D layer. Remember to call it 'mean' using the parameter `name`
    branch.add(tf.keras.layers.GlobalAveragePooling1D())
    # Add the normalizing layer using the Lambda function. Remember to call it 'out' using the parameter `name`
    branch.add(tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x)))
    
    # Define both inputs. Remember to call then 'input_1' and 'input_2' using the `name` parameter. 
    # Be mindful of the data type and size
    input1 = tf.keras.layers.Input(shape=(1,), dtype=tf.string, name='input_1')
    input2 = tf.keras.layers.Input(shape=(1,), dtype=tf.string, name='input_2')
    # Define the output of each branch of your Siamese network. Remember that both branches have the same coefficients, 
    # but they each receive different inputs.
    branch1 = branch(input1)
    branch2 = branch(input2)
    # Define the Concatenate layer. You should concatenate columns, you can fix this using the `axis`parameter. 
    # This layer is applied over the outputs of each branch of the Siamese network
    conc = tf.keras.layers.Concatenate(axis=1, name='conc_1_2')([branch1, branch2]) 
    
    ### END CODE HERE ###
    
    return tf.keras.models.Model(inputs=[input1, input2], outputs=conc, name="SiameseModel")


def TripletLossFn(v1, v2,  margin=0.25):
    """Custom Loss function.

    Args:
        v1 (numpy.ndarray or Tensor): Array with dimension (batch_size, model_dimension) associated to Q1.
        v2 (numpy.ndarray or Tensor): Array with dimension (batch_size, model_dimension) associated to Q2.
        margin (float, optional): Desired margin. Defaults to 0.25.

    Returns:
        triplet_loss (numpy.ndarray or Tensor)
    """
   
    ### START CODE HERE ###

    # use `tf.linalg.matmul` to take the dot product of the two batches. 
    # Don't forget to transpose the second argument using `transpose_b=True`
    scores = tf.linalg.matmul(v2, v1, transpose_b=True)
    # calculate new batch size and cast it as the same datatype as scores. 

    batch_size = tf.cast(tf.shape(v1)[0], scores.dtype) 
    # use `tf.linalg.diag_part` to grab the cosine similarity of all positive examples
    positive = tf.linalg.diag_part(scores)
    # subtract the diagonal from scores. You can do this by creating a diagonal matrix with the values 
    # of all positive examples using `tf.linalg.diag`
    negative_zero_on_duplicate = scores - tf.linalg.diag(positive)
    # use `tf.math.reduce_sum` on `negative_zero_on_duplicate` for `axis=1` and divide it by `(batch_size - 1)`
    mean_negative = tf.math.reduce_sum(negative_zero_on_duplicate, axis=1) / ( batch_size - 1.)
    # create a composition of two masks: 
    # the first mask to extract the diagonal elements, 
    # the second mask to extract elements in the negative_zero_on_duplicate matrix that are larger than the elements in the diagonal 
    mask_exclude_positives = tf.cast(
        (tf.eye(batch_size) == 1)|(negative_zero_on_duplicate > tf.expand_dims(positive, 1)),
                                    scores.dtype)
    # multiply `mask_exclude_positives` with 2.0 and subtract it out of `negative_zero_on_duplicate`
    negative_without_positive = negative_zero_on_duplicate - 2.0 * mask_exclude_positives
    # take the row by row `max` of `negative_without_positive`. 
    # Hint: `tf.math.reduce_max(negative_without_positive, axis = None)`
    closest_negative =tf.math.reduce_max(negative_without_positive, axis=1)
    # compute `tf.maximum` among 0.0 and `A`
    # A = subtract `positive` from `margin` and add `closest_negative` 
    triplet_loss1 = tf.maximum(mean_negative - positive + margin, 0.0)
    # compute `tf.maximum` among 0.0 and `B`
    # B = subtract `positive` from `margin` and add `mean_negative` 
    triplet_loss2 = tf.maximum(closest_negative - positive + margin, 0.0)
    # add the two losses together and take the `tf.math.reduce_sum` of it
    triplet_loss = tf.math.reduce_sum( triplet_loss2 + triplet_loss1 ) / 1
    
    ### END CODE HERE ###

    return triplet_loss

#@tf.keras.saving.register_keras_serializable(package="tripletloss_package", name="TripletLoss")
def TripletLoss(labels, out, margin=0.25):
    _, embedding_size = out.shape # get embedding size
    v1 = out[:,:int(embedding_size/2)] # Extract v1 from out
    v2 = out[:,int(embedding_size/2):] # Extract v2 from out
    return TripletLossFn(v1, v2, margin=margin)


def built_siamense_model(
            Siamese=Siamese, TripletLoss = TripletLoss, 
            text_vectorizer=None, train_Q=None,
            d_feature=128, 
            lr=0.01, 
            ):
    model = Siamese(text_vectorizer,
                    vocab_size = text_vectorizer.vocabulary_size(), #set vocab_size accordingly to the size of your vocabulary
                    d_feature = d_feature)
    # Compile the model
    
    model.compile(loss=TripletLoss,
                  optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
            )
    
    train_dataset = tf.data.Dataset.from_tensor_slices(((train_Q[0], train_Q[1]),tf.constant([1]*len(train_Q[0]))))
    batch_size = 256
    train_generator = train_dataset.shuffle(len(train_Q[0]),
                                        seed=7, 
                                        reshuffle_each_iteration=True).batch(batch_size=batch_size)
    model.fit(train_generator, epochs=1)
    tf.keras.models.save_model(model, "./models/siamense/siamense.tf", save_format="tf")     

    return model

def siamense_prediction(question1, question2, threshold=0.7, model=None, verbose=False):
    """Function for predicting if two questions are duplicates.

    Args:
        question1 (str): First question.
        question2 (str): Second question.
        threshold (float): Desired threshold.
        model (tensorflow.keras.Model): The Siamese model.
        data_generator (function): Data generator function. Defaults to data_generator.
        verbose (bool, optional): If the results should be printed out. Defaults to False.

    Returns:
        bool: True if the questions are duplicates, False otherwise.
    """
    generator = tf.data.Dataset.from_tensor_slices((([question1], [question2]),None)).batch(batch_size=1)
       
    # Call the predict method of your model and save the output into v1v2
    pred = model.predict(generator)
    # Extract v1 and v2 from the model output
    _, n_feat = pred.shape
    v1 = pred[:, :n_feat//2]
    v2 = pred[:, n_feat//2:]
    # Take the dot product to compute cos similarity of each pair of entries, v1, v2
    # Since v1 and v2 are both vectors, use the function tf.math.reduce_sum instead of tf.linalg.matmul
    d = tf.reduce_sum(tf.multiply(v1, v2)) / (tf.norm(v1) * tf.norm(v2))
    # Is d greater than the threshold?
    res = d > threshold

    result = {"cosine similarity" : d.numpy(), "question duplicates" : res.numpy()}
    
    return result
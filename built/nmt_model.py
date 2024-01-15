import tensorflow as tf
from built.softmax import softmax 

class model_nmt:
    def __init__(self, machine_vocab) -> None:
        self.Tx = 30
        self.repeator = tf.keras.layers.RepeatVector(self.Tx)
        self.concatenator = tf.keras.layers.Concatenate(axis=-1)
        self.densor1 = tf.keras.layers.Dense(10, activation = "tanh")
        self.densor2 = tf.keras.layers.Dense(1, activation = "relu")
        self.activator = tf.keras.layers.Activation(softmax, name='attention_weights') 
        # We are using a custom softmax(axis = 1) loaded in this notebook
        self. dotor = tf.keras.layers.Dot(axes = 1)
        self.n_a = 32 # number of units for the pre-attention, bi-directional LSTM's hidden state 'a'
        self.n_s = 64 # number of units for the post-attention LSTM's hidden state "s"
        self.machine_vocab = machine_vocab
        # Please note, this is the post attention LSTM cell.  
        self.post_activation_LSTM_cell = tf.keras.layers.LSTM(self.n_s, return_state = True) # Please do not modify this global variable.
        self.output_layer = tf.keras.layers.Dense(len(self.machine_vocab), activation=softmax)

    def one_step_attention(self, a, s_prev):
        """
        Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
        "alphas" and the hidden states "a" of the Bi-LSTM.
        
        Arguments:
        a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
        s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)
        
        Returns:
        context -- context vector, input of the next (post-attention) LSTM cell
        """
        
        ### START CODE HERE ###
        # Use repeator to repeat s_prev to be of shape (m, Tx, n_s) so that you can concatenate it with all hidden states "a" (≈ 1 line)
        s_prev = self.repeator(s_prev)
        # Use concatenator to concatenate a and s_prev on the last axis (≈ 1 line)
        # For grading purposes, please list 'a' first and 's_prev' second, in this order.
        concat = self.concatenator([a, s_prev])
        # Use densor1 to propagate concat through a small fully-connected neural network to compute the "intermediate energies" variable e. (≈1 lines)
        e = self.densor1(concat)
        # Use densor2 to propagate e through a small fully-connected neural network to compute the "energies" variable energies. (≈1 lines)
        energies = self.densor2(e)
        # Use "activator" on "energies" to compute the attention weights "alphas" (≈ 1 line)
        alphas = self.activator(energies)
        # Use dotor together with "alphas" and "a", in this order, to compute the context vector to be given to the next (post-attention) LSTM-cell (≈ 1 line)
        context = self.dotor([alphas, a])
        ### END CODE HERE ###
        
        return context

    def modelf(self, Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size):
        """
        Arguments:
        Tx -- length of the input sequence
        Ty -- length of the output sequence
        n_a -- hidden state size of the Bi-LSTM
        n_s -- hidden state size of the post-attention LSTM
        human_vocab_size -- size of the python dictionary "human_vocab"
        machine_vocab_size -- size of the python dictionary "machine_vocab"

        Returns:
        model -- Keras model instance
        """
        
        # Define the inputs of your model with a shape (Tx, human_vocab_size)
        # Define s0 (initial hidden state) and c0 (initial cell state)
        # for the decoder LSTM with shape (n_s,)
        X = tf.keras.layers.Input(shape=(Tx, human_vocab_size))
        # initial hidden state
        s0 = tf.keras.layers.Input(shape=(n_s,), name='s0')
        # initial cell state
        c0 = tf.keras.layers.Input(shape=(n_s,), name='c0')
        # hidden state
        s = s0
        # cell state
        c = c0
        
        # Initialize empty list of outputs
        outputs = []
        
        ### START CODE HERE ###
        
        # Step 1: Define your pre-attention Bi-LSTM. (≈ 1 line)
        a = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=n_a, return_sequences=True))(X)
        
        # Step 2: Iterate for Ty steps
        for t in range(Ty):
        
            # Step 2.A: Perform one step of the attention mechanism to get back the context vector at step t (≈ 1 line)
            context = self.one_step_attention(a=a, s_prev=s)
            
            # Step 2.B: Apply the post-attention LSTM cell to the "context" vector. (≈ 1 line)
            # Don't forget to pass: initial_state = [hidden state, cell state] 
            # Remember: s = hidden state, c = cell state
            _, s, c = self.post_activation_LSTM_cell(inputs=context, initial_state=[s, c])
            
            # Step 2.C: Apply Dense layer to the hidden state output of the post-attention LSTM (≈ 1 line)
            out = self.output_layer(inputs=s)
            
            # Step 2.D: Append "out" to the "outputs" list (≈ 1 line)
            outputs.append(out)
        
        # Step 3: Create model instance taking three inputs and returning the list of outputs. (≈ 1 line)
        model = tf.keras.models.Model(outputs=outputs, inputs=[X, s0, c0])
        
        ### END CODE HERE ###
        
        return model
    
    def built(self):
        self.len_human_vocab = 37
        self.len_machine_vocab = 11
        self.Ty = 10
        model = self.modelf(self.Tx, self.Ty, self.n_a, self.n_s, self.len_human_vocab, self.len_machine_vocab )
        opt = tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.999, learning_rate=0.005)  
        model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics = ['accuracy'])

        return model


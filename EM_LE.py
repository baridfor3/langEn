import tensorflow as tf
class EmbeddingSharedWeights(tf.keras.layers.Layer):
    """Calculates input embeddings and pre-softmax linear with shared weights."""

    def __init__(self,
                 vocab_size,
                 num_units,
                 pad_id=0,
                 mask_token=0,
                 affine=True,
                 name="embedding",
                 domain_index=[],
                 freq_domain_index=[]):
        """Specify characteristic parameters of embedding layer.
    Args:
      vocab_size: Number of tokens in the embedding.
      num_units: Dimensionality of the embedding.
      pad_id: Default 0.
      mask_id: Default 0.
      affine:Default True.
      domain_index: The domain list [[x,x,x,x],[y,y,y,y]].
                NOTE that each domain should include common tokens like [EOS], [PADDING], [SOS], etc.
    """
        super(EmbeddingSharedWeights, self).__init__(name=name)
        self.vocab_size = vocab_size
        self.num_units = num_units
        self.pad_id = pad_id
        self.mask_token = mask_token
        self.domain_flag = 0
        self.affine = affine
        self.num_heads = 8
        self.shared_weights = self.add_weight(
            shape=[self.vocab_size, self.num_units],
            dtype="float32",
            name="shared_weights",
            initializer=tf.random_normal_initializer(
                mean=0., stddev=self.num_units**-0.5))
        self.domain_index = domain_index
        self.freq_domain_index = tf.reshape(freq_domain_index, [3, -1])
        if len(self.domain_index) > 0:
            self.set_domain_bias(self.domain_index)
        self.concessions_count = 0
        self.similarity_loss = 0

    def build(self, input_shape):
        # self.shared_weights = self.em_shared_weights.get_weights()[0]
        if self.affine:
            self.affine_transformation = self.add_weight(
                shape=[self.vocab_size],
                dtype="float32",
                name="shared_weights_affline",
                initializer=tf.random_normal_initializer(
                    mean=0., stddev=self.num_units**-0.5))

        super(EmbeddingSharedWeights, self).build(input_shape)
        # self.build = True

    def set_domain_bias(self, domain):
        self.domain_bias_matrix = tf.zeros([1, self.vocab_size])
        for d in domain:
            dom = self.domain_filter(self.vocab_size, d)
            self.domain_bias_matrix = tf.concat(
                (self.domain_bias_matrix, tf.reshape(dom, [1, self.vocab_size])), 0)

    def call(
        self,
        inputs,
        linear=False,
        affine=True,
        domain_id=None,
    ):
        if linear:
            return self._linear(inputs)
        return self._embedding(inputs)

    def _embedding(
        self,
        inputs,
    ):
        embeddings = tf.gather(self.shared_weights, inputs)
        # # Scale embedding by the sqrt of the hidden size
        mask = tf.cast(tf.not_equal(inputs, 0), embeddings.dtype)
        embeddings *= tf.expand_dims(mask, -1)
        embeddings *= self.num_units**0.5
        return embeddings

    def _linear(self, inputs, domain_id=None):
        """Computes logits by running x through a linear layer.
    Args:
      x: A float32 tensor with shape [batch_size, length, num_units]
    Returns:
      float32 tensor with shape [batch_size, length, vocab_size].
    """

        # logits = tf.matmul(logits, self.projection_weights)
        #
        def __out_weight(id):
            print('Domain filtering: ' + str(id))
            # domain_id = tf.cast(id,tf.int8)
            out_weights = self.domain_filter(self.vocab_size,
                                             self.domain_index[id])
            out_weights = tf.cast(out_weights, tf.float32)
            return out_weights

        batch_size = tf.shape(input=inputs)[0]
        length = tf.shape(input=inputs)[1]
        logits = tf.reshape(inputs, [-1, self.num_units])

        logits = tf.matmul(logits, self.shared_weights, transpose_b=True)
        if self.affine:
            logits = tf.add(logits, self.affine_transformation)
        if domain_id is not None and len(self.domain_index) > 0:
            out_weights = tf.py_function(__out_weight, [domain_id],
                                         [tf.float32])
            # logits = logits + tf.expand_dims(out_weights, 0)
            logits = logits + out_weights
        re = tf.reshape(logits, [batch_size, length, self.vocab_size])
        return re

    def _LE(self, domain_id):
        """
            domain_id = [batch, seq_length, language_ids]
            e.g.,
        """

        domain = tf.gather(self.freq_domain_index, domain_id)
        # domain_id = [batch, seq_length, frequent_word_embedding_ids]

        domain = self._embedding(domain)
        # domain_id = [batch, seq_length,  frequent_word_embedding_ids, dim]
        return tf.reduce_mean(domain, [-2])

    def domain_filter(src, vocabulary, domain_index):
        """
            return [vocabulary]
        """
        updates = tf.ones(len(domain_index))
        re = tf.scatter_nd(domain_index, updates, tf.constant([vocabulary]))
        return (1. - re) * -1e9

    def get_config(self):
        # config = super(EmbeddingSharedWeights, self).get_config()
        c = {
            'vocab_size': self.vocab_size,
            'num_units': self.num_units,
            'pad_id': self.pad_id,
            'name': self.name,
        }
        # config.update(c)
        return c

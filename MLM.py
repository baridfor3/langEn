from UNIVERSAL.model import transformer
from UNIVERSAL.utils import padding_util
import tensorflow as tf
from UNIVERSAL.basic_metric import seq2seq_metric
from UNIVERSAL.block import TransformerBlock
import EM_LE


class bilingual_MLM(transformer.Transformer):
    def __init__(self,
                 vocabulary_size=40000,
                 embedding_size=512,
                 batch_size=64,
                 num_units=512,
                 num_heads=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 dropout=0.1,
                 max_seq_len=60,
                 src_sos_id=1,
                 tgt_sos_id=1,
                 src_eos_id=2,
                 tgt_eos_id=2,
                 pad_id=0,
                 mask_id=4,
                 unk_id=3,
                 label_smoothing=0.1,
                 LANG_1=1,
                 LANG_2=2, **kwargs):
        super(bilingual_MLM,
              self).__init__(vocabulary_size=vocabulary_size,
                             embedding_size=embedding_size,
                             batch_size=batch_size,
                             num_units=num_units,
                             num_heads=num_heads,
                             num_encoder_layers=num_encoder_layers,
                             num_decoder_layers=num_decoder_layers,
                             dropout=dropout,
                             max_seq_len=max_seq_len,
                             src_sos_id=src_sos_id,
                             tgt_sos_id=tgt_sos_id,
                             src_eos_id=src_eos_id,
                             tgt_eos_id=tgt_eos_id,
                             pad_id=pad_id,
                             mask_id=mask_id,
                             unk_id=unk_id,
                             label_smoothing=label_smoothing,            ffn_activation="gelu")
        ## setting NaiveSeq2Seq_model.##
        lang_1_vocabulary = [[i] for i in kwargs['lang_1_vocabulary']]
        lang_2_vocabulary = [[i] for i in kwargs['lang_2_vocabulary']]
        freq_id = kwargs['freq_id']

        self.encoder = TransformerBlock.TransformerEncoder(
            num_units,
            num_heads,
            num_encoder_layers,
            dropout,
            True,
            name="ENC",
            ffn_activation='gelu'
        )
        self.decoder = TransformerBlock.TransformerDecoder(
            num_units,
            num_heads,
            num_decoder_layers,
            dropout,
            True,
            name="DEC",
            ffn_activation='gelu'
        )
        self.LANG_1 = LANG_1
        self.LANG_2 = LANG_2
        self.set_phase(1)
        domain_index = [
            lang_1_vocabulary, lang_2_vocabulary]
        self.embedding_softmax_layer = EM_LE.EmbeddingSharedWeights(
            self.vocabulary_size, self.num_units, domain_index=domain_index, mask_token=self.MASK_ID, freq_domain_index=freq_id)
        self.O = tf.keras.layers.Dense(self.vocabulary_size)
        self.classification = tf.keras.layers.Dense(3)

        # self.seq2seq_metric = lambda x: x
        self.pre_training_metric = seq2seq_metric.MetricLayer(
            self.eos, prefix="pretraining_")

    def set_phase(self, phase=0):
        self.phase = phase
        # over write SUPER object

    def backTranslation_phase(self, data, lang_1, lang_2):
        ((permutated_x, permutated_y, x, y), ) = data
        synthetic_y = self.call(x,
                                training=False,
                                src_id=lang_1,
                                tgt_id=lang_2,
                                sos=1,
                                eos=2)
        synthetic_x = self.call(y,
                                training=False,
                                src_id=lang_2,
                                tgt_id=lang_1,
                                sos=1,
                                eos=2)
        de_real_y = tf.pad(y, [[0, 0], [1, 0]],
                           constant_values=self.sos)[:, :-1]
        de_real_x = tf.pad(x, [[0, 0], [1, 0]],
                           constant_values=self.sos)[:, :-1]
        input_src = tf.concat(
            [permutated_x, permutated_y, synthetic_x, synthetic_y], -1)
        input_tgt = tf.concat([de_real_x, de_real_y, de_real_y, de_real_x], -1)
        tgt = tf.concat([x, y, y, x], -1)
        src_lang_ids = tf.concat([
            tf.zeros_like(permutated_x, dtype=tf.int32) + lang_1,
            tf.zeros_like(permutated_y, dtype=tf.int32) + lang_2,
            tf.zeros_like(synthetic_x, dtype=tf.int32) + lang_1,
            tf.zeros_like(synthetic_y, dtype=tf.int32) + lang_2
        ], 0)
        tgt_lang_ids = tf.concat([
            tf.zeros_like(x, dtype=tf.int32) + lang_1,
            tf.zeros_like(y, dtype=tf.int32) + lang_2,
            tf.zeros_like(y, dtype=tf.int32) + lang_2,
            tf.zeros_like(x, dtype=tf.int32) + lang_1,
        ], 0)
        return input_src, input_tgt, tgt, src_lang_ids, tgt_lang_ids

    def forward_phase(self, data, lang_1=1, lang_2=2):
        ((x_input_span, x_output_span, x_label, y_input_span, y_output_span,
          y_label), ) = data
        x_input_span, y_input_span = padding_util.pad_tensors_to_same_length(
            x_input_span, y_input_span)
        x_output_span, y_output_span = padding_util.pad_tensors_to_same_length(
            x_output_span, y_output_span)
        x_label, y_label = padding_util.pad_tensors_to_same_length(
            x_label, y_label)
        input_src = tf.concat([x_input_span, y_input_span], 0)
        output_tgt = tf.concat([x_output_span, y_output_span], 0)
        tgt = tf.concat([x_label, y_label], 0)
        src_lang_ids = tf.concat([
            tf.zeros_like(x_input_span, dtype=tf.int32) + lang_1,
            tf.zeros_like(y_input_span, dtype=tf.int32) + lang_2
        ], 0)
        tgt_lang_ids = tf.concat([
            tf.zeros_like(x_output_span, dtype=tf.int32) + lang_1,
            tf.zeros_like(y_output_span, dtype=tf.int32) + lang_2
        ], 0)
        return input_src, output_tgt, tgt, src_lang_ids, tgt_lang_ids

    def call(self, inputs, training, **kwargs):
        if training:
            src_id = kwargs["src_id"]
            tgt_id = kwargs["tgt_id"]
            tgt_label = kwargs["tgt_label"]

            src, tgt = inputs[0], inputs[1]
            attention_bias, decoder_self_attention_bias, decoder_padding = self.pre_processing(
                src, tgt)
            src_id *= tf.cast(tf.not_equal(src, 0), tf.int32)
            tgt_id *= tf.cast(tf.not_equal(tgt, 0), tf.int32)
            src = self.embedding_softmax_layer(src)
            tgt = self.embedding_softmax_layer(tgt)
            src_id = self.embedding_softmax_layer._LE(src_id)
            src += src_id
            tgt_id = self.embedding_softmax_layer._LE(tgt_id)
            tgt += tgt_id
            if "encoder" in kwargs:
                enc = self.encoding(src, training=training,
                                    attention_bias=attention_bias)
                if self.phase == "4":
                    logits = self.classification(enc)
                else:
                    logits = enc
                return logits

            else:

                dec = self.forward(
                    src,
                    tgt,
                    training=training,
                    attention_bias=attention_bias,
                    decoder_self_attention_bias=decoder_self_attention_bias,
                    decoder_padding=decoder_padding)
                logits = self.O(dec)
            return logits
        else:
            src_id = kwargs["src_id"]
            tgt_id = kwargs["tgt_id"]
            sos = kwargs["sos"]
            eos = kwargs["eos"]
            src = inputs
            src = self.embedding_softmax_layer(src)
            max_decode_length = self.max_seq_len
            autoregressive_fn = self.autoregressive_fn(
                max_decode_length, self.lang_emebedding(src_id), tgt_id)
            cache, batch_size = self.prepare_cache(src, sos)
            re, score = self.predict(batch_size,
                                     autoregressive_fn,
                                     eos_id=eos,
                                     cache=cache)
            return re

    def train_step(self, data):
        if self.phase == 4:
            input_src, output_tgt, tgt_label, src_lang_ids, tgt_lang_ids = self.backTranslation_phase(
                data, lang_1=self.LANG_1, lang_2=self.LANG_2)
        else:
            input_src, output_tgt, tgt_label, src_lang_ids, tgt_lang_ids = self.forward_phase(
                data, lang_1=self.LANG_1, lang_2=self.LANG_2)
        mask = tf.cast(tf.equal(input_src, self.MASK_ID), tf.int32) + tf.cast(tf.equal(input_src,
                                                                                       self.SRC_SOS_ID), tf.int32) + tf.cast(tf.equal(input_src, self.SRC_EOS_ID), tf.int32)
        tgt_pre = tgt_label * mask
        _ = self.seq2seq_training(input_src,
                                  output_tgt,
                                  tgt_pre,
                                  src_id=src_lang_ids,
                                  tgt_id=tgt_lang_ids,
                                  tgt_label=tgt_label)
        return {m.name: m.result() for m in self.metrics}

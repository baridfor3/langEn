##########
from UNIVERSAL.data_and_corpus import offline_corpus, data_manager, dataset_preprocessing
import configuration
from UNIVERSAL.basic_optimizer import learning_rate_op, optimizer_op
from UNIVERSAL.training_and_learning import callback_training
from UNIVERSAL.data_and_corpus import offline_vocabulary
import MLM
import os
import masking_schema
import tensorflow as tf
cwd = os.getcwd()
# offline dataset offline=[[src_path],[tgt_path]]
offline = [
    [
        "Tokenized.news.2008.de.shuffled_filter",
    ],
    [
        "Tokenized.news.2007.en.shuffled_filter",
    ]
]


def tf_output(x_input_span, x_output_span, x_label, y_input_span,
              y_output_span, y_label):
    return ((x_input_span, x_output_span, x_label, y_input_span,
            y_output_span, y_label), )


def preprocessed_dataset(schema):
    training_samples = offline_corpus.offline(offline)
    # path to the BPE vocabulary
    dataManager = data_manager.DatasetManager(
        cwd + '/../UNIVERSAL/vocabulary/DeEn_60000/', training_samples)
    dataset = dataManager.get_raw_train_dataset()
    if schema == 'MASS':
        tf_encode_schema = masking_schema.MASS_with_EOS_entailment
    if schema == 'XLM':
        tf_encode_schema = masking_schema.XLM_with_EOS_entailment
    else:
        tf_encode_schema = masking_schema.MASS_with_EOS_entailment


    preprocessed_dataset = dataset_preprocessing.prepare_training_input(
        dataset,
        configuration.parameters['batch_size'],
        configuration.parameters['max_sequence_length'],
        # for masking shcema. AKA: MASS, XLM, mBART etc.
        tf_encode= tf_encode_schema,
    )
    return preprocessed_dataset, dataManager


def optimizer():
    return optimizer_op.AdamWeightDecay(
        # weight_decay_rate=0.001,
        exclude_from_weight_decay=["layer_norm", "bias"], )


def callbacks():
    lr_schedual = learning_rate_op.LearningRateScheduler(
        learning_rate_op.LearningRateFn(
            learning_rate=configuration.parameters["lr"],
            hidden_size=configuration.parameters["num_units"],
            warmup_steps=configuration.parameters["learning_warmup"]), 0)
    return callback_training.get_callbacks(cwd, lr_schedual)


def trainer():
    bi_model = MLM.bilingual_MLM(
        vocabulary_size=configuration.parameters["vocabulary_size"],
        embedding_size=configuration.parameters["embedding_size"],
        batch_size=configuration.parameters["batch_size"],
        num_units=configuration.parameters["num_units"],
        num_heads=configuration.parameters["num_heads"],
        num_encoder_layers=configuration.parameters["num_encoder_layers"],
        num_decoder_layers=configuration.parameters["num_decoder_layers"],
        dropout=configuration.parameters["dropout"],
        max_seq_len=configuration.parameters["max_sequence_length"],
        LANG_1=configuration.parameters["LANG_1"],
        LANG_2=configuration.parameters["LANG_2"],
        # optional for mononlingual vocabulary
        lang_1_vocabulary=offline_vocabulary.read_offine_vocabulary(
            "./UNIVERSAL/vocabulary/DeEn_60000/vocab_monolingual_De60000.vocab"),
        lang_2_vocabulary=offline_vocabulary.read_offine_vocabulary(
            "./UNIVERSAL/vocabulary/DeEn_60000/vocab_monolingual_En60000.vocab"),
        freq_id=freq_id_for_domain(100)
    )
    return bi_model


# read the most frequent tokens from statistic
def freq_id_for_domain(top_k):
    domain_0 = []
    with open("/home/vivalavida/workspace/alpha/UNIVERSAL/vocabulary/DeEn_60000/statistic_monolingual_De60000.statistic", "r") as f:
        for k, v in enumerate(f.readlines()):
            if k > top_k:
                break
            domain_0.append(int(v.split("@")[0]))
    domain_1 = []
    with open("/home/vivalavida/workspace/alpha/UNIVERSAL/vocabulary/DeEn_60000/statistic_monolingual_En60000.statistic", "r") as f:
        for k, v in enumerate(f.readlines()):
            if k > top_k:
                break
            domain_1.append(int(v.split("@")[0]))
    cross_domain = set.intersection(set(domain_0), set(domain_1))
    domain_0 = list(set(domain_0) - cross_domain)
    domain_1 = list(set(domain_1) - cross_domain)
    return [[0 for i in range(0, len(domain_0))], domain_0, domain_1]

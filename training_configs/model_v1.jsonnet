{
    dataset_reader: {
        type: "commonlit_reader",
        tokenizer: "whitespace",
    },
    vocabulary: {
        "pretrained_files": {
            "tokens": "../input/glove-vec/glove.6B.100d.word2vec",
        },
    },
    train_data_path: "./processed_train.csv",
    validation_data_path: "./processed_valid.csv",
    model: {
        type: "naive",
        excerpt_embedder: {
            type: "basic",
            token_embedders: {
                tokens: {
                    embedding_dim: 100,
                    pretrained_file: "../input/glove-vec/glove.6B.100d.word2vec"
                }
            }
        },
        excerpt_encoder: {
            type: "lstm",
            input_size: 100,
            hidden_size: 50,
        }
    },
    trainer: {
        num_epochs: 30,
        optimizer: {
            type: "adam",
            lr: 1e-5
        },
        validation_metric: "-loss"
    },
    data_loader: {
        batch_size: 16,
        shuffle: true
    }
}
